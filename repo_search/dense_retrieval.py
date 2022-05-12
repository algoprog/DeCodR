import json
import logging
import math

from collections import defaultdict
from typing import List, Union
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers.models import Pooling, Transformer
from sentence_transformers.util import dot_score
from torch.utils.data import DataLoader

from reranker import ReRanker
from vector_index import VectorIndex


class InputExample:
    def __init__(self, texts: List[str], label: Union[int, float] = 0):
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, text: {}".format(str(self.label), self.texts[0])


class DenseEncoder:
    def __init__(self, model_path=None):
        if model_path is not None:
            self.load_model(pretrained_model_path=model_path)

    def load_model(self, pretrained_model_path=None, model_name="distilroberta-base"):
        logging.info("Loading model weights...")

        if pretrained_model_path is None:
            word_embedding_model = Transformer(model_name, max_seq_length=512)
            pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=False,
                                    pooling_mode_cls_token=True,
                                    pooling_mode_max_tokens=False)
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        else:
            self.model = SentenceTransformer(pretrained_model_path)

        logging.info("Loaded model weights.")

    def train(self,
              data_path="data",
              model_name="distilroberta-base",
              pretrained_model_path=None,
              sep_token="</s>",
              output_path="weights",
              epochs=10,
              evaluation_steps=1000,
              warmup_steps=100,
              batch_size=64,
              ):
        self.load_model(pretrained_model_path=pretrained_model_path,
                        model_name=model_name)

        logging.info("Loading dataset...")

        corpus = {}  # Our whole corpus, pid => passage

        # Read passages
        with open(f"{data_path}/docs.jsonl", encoding="utf8") as f:
            for line in f:
                d = json.loads(line.rstrip("\n"))
                name = d["name"]
                description = d["description"] if d["description"] is not None else ""
                readme = d["readme"] if d["readme"] is not None else ""
                corpus[d["id"]] = sep_token.join([name, description, readme])

        train_examples = []
        with open(f"{data_path}/queries_train.jsonl", encoding="utf8") as f:
            for line in f:
                d = json.loads(line.rstrip("\n"))
                for doc_id in d["docs"]:
                    train_examples.append(InputExample(texts=[d["query"], corpus[doc_id]], label=1))

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        train_loss = MultipleNegativesRankingLoss(self.model, similarity_fct=dot_score)

        dev_corpus = {}
        dev_queries = {}  # Our dev queries. qid => query
        dev_rel_docs = defaultdict(lambda: set())  # Mapping qid => set with relevant pids
        dev_doc_ids = set()

        with open(f"{data_path}/queries_test.jsonl", encoding="utf8") as f:
            for line in f:
                d = json.loads(line.rstrip("\n"))
                dev_queries[d["qid"]] = d["query"]
                for doc_id in d["docs"]:
                    dev_doc_ids.add(doc_id)
                    dev_rel_docs[d["qid"]].add(doc_id)
        for doc_id in dev_doc_ids:
            dev_corpus[doc_id] = corpus[doc_id]

        logging.info("Train examples: {}".format(len(train_examples)))
        logging.info("Queries (train/dev): {}".format(len(dev_queries)))
        logging.info("Corpus: {}".format(len(corpus)))
        logging.info("Corpus (dev): {}".format(len(dev_corpus)))

        ir_evaluator = InformationRetrievalEvaluator(dev_queries, dev_corpus, dev_rel_docs,
                                                     show_progress_bar=True,
                                                     corpus_chunk_size=1000,
                                                     mrr_at_k=[10, 100],
                                                     ndcg_at_k=[10, 100],
                                                     accuracy_at_k=[1, 3, 5, 10, 100],
                                                     precision_recall_at_k=[1, 3, 5, 10, 100],
                                                     name="dev")

        self.model.evaluate(ir_evaluator)
        exit()

        logging.info("Training model...")

        # Train the model
        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                       evaluator=ir_evaluator,
                       epochs=epochs,
                       evaluation_steps=evaluation_steps,
                       warmup_steps=warmup_steps,
                       output_path=output_path)

    def index_documents(self, docs, ids, use_gpu=True, dim=768):
        self.index = VectorIndex(dim)
        self.ids = ids
        vectors = self.model.encode(docs)
        for v in vectors:
            self.index.add(v)
        logging.info("Building examples index...")
        self.index.build(use_gpu=use_gpu)

    def search(self, query, topk=100):
        v = self.model.encode([query], show_progress_bar=False)
        ids, scores = self.index.search(vectors=v, k=topk)
        ids = ids[0]
        scores = scores[0]
        results = [(self.ids[id_], score) for id_, score in zip(ids, scores)]
        return results


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    # training example
    #de = DenseEncoder()
    #de.train(pretrained_model_path="weights", output_path="weights")
    
    # search example
    retriever = DenseEncoder("weights_pretrained")
    #retriever.interactive_demo()

    from evaluate import evaluate
    evaluate(retriever, split="dev", rerank=True)
    evaluate(retriever, split="test", rerank=True)

    #retriever = DenseEncoder("weights")
    #evaluate(retriever, split="dev")
    #evaluate(retriever, split="test")
