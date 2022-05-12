import os
import torch
import json
import logging
import random
import re
import math
import wandb

from torch import nn
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from collections import defaultdict
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer, AutoTokenizer
from syntok.tokenizer import Tokenizer
from collections import defaultdict
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from sparse_retrieval import SparseRetriever

os.environ["NCCL_DEBUG"] = "INFO"
random.seed(42)


def convert_example_to_features(
        text: str,
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        use_segment_ids=False,
        cls_token_segment_id=0,
        sep_token="[SEP]",
        sep_token_extra=True,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=11,
        mask_padding_with_zero=True,
):
    tokens_ = tokenizer.tokenize(text)

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = tokenizer.num_special_tokens_to_add()
    if len(tokens_) > max_seq_length - special_tokens_count:
        tokens_ = tokens_[: (max_seq_length - special_tokens_count)]

    tokens = []
    segment_ids = []
    segment_id = 0
    for word in tokens_:
        segment_ids.append(segment_id)
        if word == sep_token:
            tokens.append(sep_token)
            segment_id += 1
        else:
            tokens.append(word)

    tokens += [sep_token]
    segment_ids.append(segment_id)

    if sep_token_extra:
        # roberta uses an extra separator b/w pairs of sentences
        tokens += [sep_token]
        segment_ids.append(segment_id)

    if cls_token_at_end:
        tokens += [cls_token]
        segment_ids += [cls_token_segment_id]
    else:
        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
    else:
        input_ids += [pad_token] * padding_length
        attention_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    if "token_type_ids" not in tokenizer.model_input_names:
        segment_ids = None

    if use_segment_ids:
        return input_ids, attention_mask, segment_ids
    else:
        return input_ids, attention_mask


class PairwiseRankingDataset:
    def __init__(
            self,
            queries_dict,
            documents_dict,
            queries_ids,
            positives_ids,
            negatives_ids,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            max_seq_length: Optional[int] = None,
            num_negatives=8,
            queries_per_batch=8,
            epochs=3
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.model_type = model_type
        self.num_negatives = num_negatives
        self.queries_per_batch = queries_per_batch
        self.epochs = epochs
        self.epoch = 0

        self.queries_dict = queries_dict
        self.documents_dict = documents_dict

        self.queries_ids = queries_ids
        self.qid_index = 0
        self.qids_pos_index = []

        self.positives_ids = positives_ids
        self.negatives_ids = negatives_ids

        self.total_examples = 0
        for pos in positives_ids:
            self.total_examples += len(pos)
        for neg in negatives_ids:
            self.total_examples += len(neg)

        self.shuffle_dataset()

    def shuffle_dataset(self):
        self.epoch += 1
        self.qid_index = 0
        logging.info("Shuffling dataset...")
        random.shuffle(self.queries_ids)
        self.qids_pos_index = [-1] * len(self.queries_ids)
        for i in range(len(self.queries_ids)):
            random.shuffle(self.positives_ids[i])
            random.shuffle(self.negatives_ids[i])

    def get_next_rankset(self):
        # get next query positive
        self.qids_pos_index[self.qid_index] += 1
        # query positives exhausted
        if self.qids_pos_index[self.qid_index] == len(self.positives_ids[self.queries_ids[self.qid_index]]):
            # get next query
            self.qid_index += 1
            # queries exhausted
            if self.qid_index == len(self.queries_ids):
                self.shuffle_dataset()

        negatives = self.negatives_ids[self.queries_ids[self.qid_index]]

        return self.queries_ids[self.qid_index], \
               self.positives_ids[self.queries_ids[self.qid_index]][self.qids_pos_index[self.qid_index]], \
               random.sample(negatives, min(self.num_negatives, len(negatives)))

    def batch_generator(self):
        while self.epoch <= self.epochs:
            batch = []
            for _ in range(self.queries_per_batch):
                query_id, positive_id, negatives_ids = self.get_next_rankset()
                if self.epoch > self.epochs:
                    return

                query = self.queries_dict[query_id]

                positive_text = "{}{}{}".format(query, self.tokenizer.sep_token, self.documents_dict[positive_id])
                pos_features = convert_example_to_features(
                    positive_text,
                    self.max_seq_length,
                    self.tokenizer,
                    cls_token_at_end=bool(self.model_type in ["xlnet"]),
                    cls_token=self.tokenizer.cls_token,
                    cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                    sep_token=self.tokenizer.sep_token,
                    sep_token_extra=False,
                    pad_on_left=bool(self.tokenizer.padding_side == "left"),
                    pad_token=self.tokenizer.pad_token_id,
                    pad_token_segment_id=self.tokenizer.pad_token_type_id)

                for negative_id in negatives_ids:
                    negative_text = "{}{}{}".format(query, self.tokenizer.sep_token, self.documents_dict[negative_id])
                    neg_features = convert_example_to_features(
                        negative_text,
                        self.max_seq_length,
                        self.tokenizer,
                        cls_token_at_end=bool(self.model_type in ["xlnet"]),
                        cls_token=self.tokenizer.cls_token,
                        cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
                        sep_token=self.tokenizer.sep_token,
                        sep_token_extra=False,
                        pad_on_left=bool(self.tokenizer.padding_side == "left"),
                        pad_token=self.tokenizer.pad_token_id,
                        pad_token_segment_id=self.tokenizer.pad_token_type_id)
                    features = pos_features + neg_features
                    batch.append(features)

            yield collate_batch(batch)


class RankingDataset(Dataset):
    def __init__(
            self,
            queries_dict,
            documents_dict,
            queries_ids,
            positives_ids,
            tokenizer: PreTrainedTokenizer,
            model_type: str,
            max_seq_length,
            negatives_ids=None,
            limit=0
    ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.model_type = model_type
        self.limit = limit

        self.queries_dict = queries_dict
        self.documents_dict = documents_dict

        self.examples = []

        if negatives_ids is not None:
            for i, query_id in enumerate(queries_ids):
                for positive_id in positives_ids[i]:
                    self.examples.append((query_id, positive_id, 1))
                for negative_id in negatives_ids[i]:
                    self.examples.append((query_id, negative_id, 0))
        else:
            for positive_id in positives_ids:
                self.examples.append((queries_ids, positive_id))

    def __len__(self):
        return len(self.examples) if self.limit == 0 else min(self.limit, len(self.examples))

    def __getitem__(self, i):
        example = self.examples[i]

        text = "{}{}{}".format(self.queries_dict[example[0]], self.tokenizer.sep_token, self.documents_dict[example[1]])

        features = convert_example_to_features(
            text,
            self.max_seq_length,
            self.tokenizer,
            cls_token_at_end=bool(self.model_type in ["xlnet"]),
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.model_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=bool(self.tokenizer.padding_side == "left"),
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id)

        if len(example) == 3:
            return features + (example[0], example[2],)  # input ids, mask, qid, label
        else:
            return features  # input ids, mask


def collate_batch(batch, all_features=False):
    num_features = len(batch[0])
    coll_batch = [[] for _ in range(num_features)]

    for sample in batch:
        for i, x in enumerate(sample):
            coll_batch[i].append(x)

    for i in range(num_features):
        if all_features or isinstance(coll_batch[i][0], list):
            t = torch.tensor(coll_batch[i]).to(torch.device("cuda"))
            coll_batch[i] = t

    return coll_batch


class ReRanker(nn.Module):
    def __init__(self,
                 model_path=None,
                 model_type="distilroberta-base",
                 use_gpu=True,
                 parallel=False,
                 debug=False,
                 max_seq_length=400):
        super(ReRanker, self).__init__()

        self.model_type = model_type

        configuration = AutoConfig.from_pretrained(self.model_type)
        if model_path is None:
            self.bert = AutoModel.from_pretrained(self.model_type)
        else:
            self.bert = AutoModel.from_config(configuration)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_type)
        self.hidden_dim = configuration.hidden_size
        self.max_seq_length = max_seq_length

        self.score = nn.Linear(self.hidden_dim, 1)

        if parallel:
            self.bert = DataParallel(self.bert)

        if model_path is not None:
            sdict = torch.load(os.path.join(model_path, "model.state_dict"), map_location=lambda storage, loc: storage)
            self.load_state_dict(sdict, strict=False)

        self.device = torch.device("cuda") if use_gpu else torch.device("cpu")

        self.to(self.device)
        self.debug = debug

    def pairwise_loss(self, batch):
        pos_logits = self.forward(batch[0], batch[1])
        neg_logits = self.forward(batch[2], batch[3])
        loss = torch.mean(torch.log(1 + torch.exp(-torch.sub(pos_logits, neg_logits))), dim=0)
        return loss

    """
    def pointwise_loss(self, batch):
        loss_fn = nn.BCELoss()
        scores = torch.sigmoid(self.forward(batch[0], batch[1]))
        loss = loss_fn(scores, batch[2].view(-1, 1))
        return loss
    """

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        cls = self.bert(input_ids=input_ids, attention_mask=attention_mask)[1]
        scores = self.score(cls)
        return scores

    def score_documents(self,
                        queries_dict,
                        documents_dict,
                        queries_ids,
                        document_ids,
                        batch_size=8):
        dataset = RankingDataset(queries_dict=queries_dict,
                                 documents_dict=documents_dict,
                                 queries_ids=queries_ids,
                                 positives_ids=document_ids,
                                 negatives_ids=None,
                                 max_seq_length=self.max_seq_length,
                                 model_type=self.model_type,
                                 tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=collate_batch)
        scores = []
        for input_ids, attention_mask in dataloader:
            scores_ = self.forward(input_ids, attention_mask).data.cpu().numpy()
            # scores_ = [s[0] for s in scores_]
            scores.extend(scores_)
        return scores

    def load_model(self, sdict):
        self.load_state_dict(sdict)
        self.to(self.device)

    def save_model(self, output_path):
        model_name = 'model.state_dict'
        opath = os.path.join(output_path, model_name)
        torch.save(self.state_dict(), opath)


def calculate_metrics(model, dataloader):
    model.eval()
    scores = defaultdict(lambda: [])
    logging.info("Running evaluation...")
    for input_ids, attention_mask, qids, labels in tqdm(dataloader, position=1):
        scores_ = model.forward(input_ids, attention_mask).data.cpu().numpy()
        for i, score in enumerate(scores_):
            r = random.random()
            scores[qids[i]].append((score, labels[i], r))

    avg_ndcg = 0
    avg_len = 0
    mrr = 0
    for _, scores_ in scores.items():
        scores_ = sorted(scores_, key=lambda x: x[0], reverse=True)
        relevant = 0
        dcg = 0
        mr = 0
        avg_len += len(scores_)
        for i, (score, label, rscore) in enumerate(scores_):
            if label == 1:
                relevant += 1
                dcg += 1 / math.log(2 + i)
                if mr == 0:
                    mr = 1 / (i + 1)
        idcg = 0
        for i in range(relevant):
            idcg += 1 / math.log(2 + i)
        ndcg = dcg / idcg
        avg_ndcg += ndcg
        mrr += mr

    total_queries = len(scores.keys())

    avg_ndcg /= total_queries
    mrr /= total_queries
    avg_len /= total_queries

    model.train()

    return avg_ndcg, mrr


def train(output_path="weights_ranker",
          model_path=None,
          model_type="distilroberta-base",
          sampling_index_path="sampling_index",
          queries_per_batch=8,
          num_negatives=4,
          eval_batch_size=8,
          max_eval_triplets=2000,
          lr=1e-5,
          accumulation_steps=1,
          warmup_steps=1000,
          max_seq_length=512,
          epochs=3,
          eval_steps=1000,
          log_steps=10,
          data_path="data",
          bm25_negatives_ratio=1.0,
          random_negatives_ratio=1.0,
          use_gpu=True,
          parallel=True,
          fp16=False,
          wandb_project="re-ranking",
          wandb_user=None):
    if not os.path.isdir(output_path):
        os.mkdir(output_path)

    train_batch_size = queries_per_batch * num_negatives

    if wandb_user is not None:
        wandb.init(project=wandb_project, entity=wandb_user)
        wandb.config = {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": train_batch_size
        }

    logging.info("Loading model...")
    model = ReRanker(model_type=model_type,
                     model_path=model_path,
                     use_gpu=use_gpu,
                     parallel=parallel,
                     max_seq_length=max_seq_length)

    logging.info("Loading corpus...")
    queries_dict = {}
    with open(f"{data_path}/queries.jsonl") as f:
        for line in f:
            d = json.loads(line.rstrip("\n"))
            queries_dict[d["qid"]] = d["query"]

    documents_dict = {}
    with open(f"{data_path}/docs.jsonl") as f:
        for line in f:
            d = json.loads(line.rstrip("\n"))
            name = d["name"]
            description = d["description"] if d["description"] is not None else ""
            readme = d["readme"] if d["readme"] is not None else ""
            documents_dict[d["id"]] = f" {model.tokenizer.sep_token} ".join([name, description, readme])
    doc_ids = list(documents_dict.keys())

    sparse_retriever = SparseRetriever(path=sampling_index_path, reset=False)
    if bm25_negatives_ratio > 0:
        logging.info("Indexing corpus...")
        #sparse_retriever.index_documents(documents=list(documents_dict.values()), ids=list(documents_dict.keys()))

    logging.info("Preparing train/dev/test triples...")
    queries_ids = [[], [], []]
    positives_ids = [[], [], []]
    negatives_ids = [[], [], []]
    for i, split in enumerate(["train", "dev", "test"]):
        avg_pos = 0
        avg_neg = 0
        with open(f"{data_path}/qrels_{split}.jsonl") as f:
            for line in tqdm(f):
                d = json.loads(line.rstrip("\n"))

                queries_ids[i].append(d["qid"])

                pos_ids = d["docs"]
                positives_ids[i].append(pos_ids)

                avg_pos += len(pos_ids)

                if bm25_negatives_ratio > 0 or split != "train":
                    if split != "train":
                        bm25_neg_count = max(len(pos_ids), 100)
                    else:
                        bm25_neg_count = int(bm25_negatives_ratio * len(pos_ids))
                    bm25_neg_ids = sparse_retriever.search(query=queries_dict[d["qid"]], topk=1000)
                    bm25_neg_ids = [x[0] for x in bm25_neg_ids]
                    if len(bm25_neg_ids) > 0:
                        bm25_neg_ids = random.choices(bm25_neg_ids, k=min(len(bm25_neg_ids), bm25_neg_count))
                else:
                    bm25_neg_ids = []

                if split == "train":
                    random_neg_count = int(random_negatives_ratio * len(pos_ids))
                    random_neg_ids = random.choices(doc_ids, k=random_neg_count)
                else:
                    random_neg_ids = []

                neg_ids = list(set(bm25_neg_ids).union(set(random_neg_ids)).difference(set(pos_ids)))
                negatives_ids[i].append(neg_ids)

                avg_neg += len(neg_ids)

        avg_pos /= len(queries_ids[i])
        avg_neg /= len(queries_ids[i])
        print(avg_pos, avg_neg)

    train_dataset = PairwiseRankingDataset(queries_dict=queries_dict,
                                           documents_dict=documents_dict,
                                           queries_ids=queries_ids[0],
                                           positives_ids=positives_ids[0],
                                           negatives_ids=negatives_ids[0],
                                           max_seq_length=max_seq_length,
                                           model_type=model_type,
                                           tokenizer=model.tokenizer,
                                           num_negatives=num_negatives,
                                           queries_per_batch=queries_per_batch)
    train_dataloader = train_dataset.batch_generator()

    dev_dataset = RankingDataset(queries_dict=queries_dict,
                                 documents_dict=documents_dict,
                                 queries_ids=queries_ids[1],
                                 positives_ids=positives_ids[1],
                                 negatives_ids=negatives_ids[1],
                                 max_seq_length=max_seq_length,
                                 model_type=model_type,
                                 tokenizer=model.tokenizer,
                                 limit=max_eval_triplets)
    dev_dataloader = DataLoader(dataset=dev_dataset,
                                batch_size=eval_batch_size,
                                shuffle=False,
                                collate_fn=collate_batch)

    test_dataset = RankingDataset(queries_dict=queries_dict,
                                  documents_dict=documents_dict,
                                  queries_ids=queries_ids[2],
                                  positives_ids=positives_ids[2],
                                  negatives_ids=negatives_ids[2],
                                  max_seq_length=max_seq_length,
                                  model_type=model_type,
                                  tokenizer=model.tokenizer,
                                  limit=max_eval_triplets)
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=eval_batch_size,
                                 shuffle=False,
                                 collate_fn=collate_batch)

    total_examples = train_dataset.total_examples

    logging.info("Training model...")

    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-6, correct_bias=False)
    total_steps = math.ceil(total_examples / (train_batch_size * accumulation_steps)) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    writer = SummaryWriter()

    best_ndcg = 0
    steps = 0
    accumulated_steps = 0
    running_loss = 0.0
    scaler = GradScaler()
    for epoch in range(epochs):
        iterator = tqdm(train_dataloader, position=0)
        for batch in iterator:
            if fp16:
                with autocast():
                    loss = model.pointwise_loss(batch)
                scaler.scale(loss).backward()
            else:
                loss = model.pairwise_loss(batch)
                loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if ((steps + 1) % accumulation_steps == 0) or (steps + 1 == total_steps):
                batch_loss_value = loss.item()
                running_loss += batch_loss_value

                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                accumulated_steps += 1

                if accumulated_steps % eval_steps == 0:
                    with torch.no_grad():
                        logging.info("Calculating dev_ndcg...")
                        dev_ndcg, dev_mrr = calculate_metrics(model, dev_dataloader)
                        #logging.info("Calculating test_ndcg...")
                        #test_ndcg, test_mrr = calculate_metrics(model, test_dataloader)

                    writer.add_scalar("dev_ndcg", dev_ndcg, accumulated_steps)
                    writer.add_scalar("dev_mrr", dev_mrr, accumulated_steps)
                    #writer.add_scalar("test_ndcg", test_ndcg, accumulated_steps)
                    #writer.add_scalar("test_mrr", test_mrr, accumulated_steps)

                    if wandb_user is not None:
                        wandb.log({"dev_ndcg": dev_ndcg})
                        wandb.log({"dev_mrr": dev_mrr})
                        #wandb.log({"test_ndcg": test_ndcg})
                        #wandb.log({"test_mrr": test_mrr})

                    if dev_ndcg > best_ndcg:
                        logging.info("ndcg improved from {} to {}, saving model weights...".format(best_ndcg, dev_ndcg))
                        best_ndcg = dev_ndcg
                        model.save_model(output_path=output_path)

                if accumulated_steps % log_steps == 0:
                    writer.add_scalar("loss", running_loss / log_steps, accumulated_steps)
                    if wandb_user is not None:
                        wandb.log({"loss": running_loss / log_steps})
                    running_loss = 0.0

                iterator.set_description("loss: {}, acc_steps: {}/{}".format(batch_loss_value,
                                                                             accumulated_steps,
                                                                             total_steps))

            steps += 1


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    train(output_path="weights_ranker",
          #model_path="weights_ranker",
          model_type="distilroberta-base",
          queries_per_batch=8,
          num_negatives=4,
          max_eval_triplets=10000,
          eval_batch_size=8,
          lr=1e-5,
          accumulation_steps=2,
          warmup_steps=100,
          max_seq_length=300,
          epochs=3,
          eval_steps=1000,
          log_steps=10,
          data_path="data",
          bm25_negatives_ratio=1.0,
          random_negatives_ratio=1.0,
          use_gpu=True,
          parallel=True,
          fp16=False,
          wandb_project="github-reranking",
          wandb_user="algoprog")
