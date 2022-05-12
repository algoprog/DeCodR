import sys

sys.path += ['../']
import torch
import hydra
import wandb
import os

from torch import nn
from pathlib import Path

from utils import EmbeddingMemoryQueue, seed_everything
from pytorch_metric_learning import losses
from validation import Validator, default_processing_fn, processing_before_indexing_fn
from args import TrainingConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from hydra.core.config_store import ConfigStore

from models import EmbeddingModel
from typing import Callable, List, Optional, Tuple, Any, Dict, Union
from ir_dataset_package.dataset import TokenizedQrelPairDataset
from ir_dataset_package.collator import TokenizedQrelPairCollator


cs = ConfigStore.instance()
cs.store(name='config', node=TrainingConfig)


def run(cfg: TrainingConfig):
    seed_everything(1234)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device {device}')

    log_name = f'{cfg.train_args.name_prefix}_{cfg.train_args.version}'
    save_dir = os.path.join(cfg.train_args.save_dir, log_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if cfg.train_args.log:
        wandb.init(
            project='code-search',
            entity='jfkback',
            settings=wandb.Settings(start_method='fork'),
            name=log_name,
            mode='online',
            notes=cfg.train_args.notes,
            tags=cfg.train_args.tags,
        )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_args.tokenizer_name_or_path)

    dataset_train = TokenizedQrelPairDataset(
        tokenizer,
        collection_path=cfg.data_args.collection_path,
        query_path=cfg.data_args.query_path,
        qrels_path=cfg.data_args.qrels_path,
        cache_dir=cfg.data_args.cache_dir,
        lazy_tokenize=cfg.data_args.lazy_tokenize,
        query_max_length=cfg.data_args.query_max_length,
        passage_max_length=cfg.data_args.passage_max_length,
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=cfg.data_args.batch_size,
        num_workers=cfg.data_args.num_workers,
        shuffle=True,
        drop_last=True,
        collate_fn=TokenizedQrelPairCollator(tokenizer, return_tensors='pt'),
    )

    model = EmbeddingModel(cfg.model_args.model_name_or_path,)

    if cfg.model_args.load_checkpoint_path is not None:
        state_dict = torch.load(
            cfg.model_args.load_checkpoint_path, map_location=torch.device('cpu')
        )
        model.load_state_dict(state_dict, strict=False)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    optimizer = hydra.utils.instantiate(
        cfg.train_args.optimizer, model.parameters(), _convert_='partial',
    )

    if cfg.train_args.lr_scheduler is not None:
        scheduler = hydra.utils.instantiate(
            cfg.train_args.lr_scheduler, optimizer, _convert_='partial',
        )

    gradient_accumulation_steps = cfg.train_args.gradient_accumulation_steps

    model.train()
    

    loss_fn = losses.NTXentLoss(temperature=1)

    optimizer.zero_grad()

    validator = Validator(
        tokenizer,
        cfg.data_args.val_collection_path,
        cfg.data_args.val_query_path,
        cfg.data_args.val_qrels_path,
        cfg.data_args.cache_dir,
        post_processing_fn=default_processing_fn,
        processing_before_indexing_fn=processing_before_indexing_fn,
        collection_max_length=cfg.data_args.query_max_length,
        query_max_length=cfg.data_args.passage_max_length,
    )

    memory_queue = EmbeddingMemoryQueue(cfg.model_args.embedding_dim)

    step = 0
    running_loss = 0
    for epoch in range(cfg.train_args.max_epochs):
        print(f'Epoch {epoch}, dataloader length {len(dataloader_train)}')
        for queries, passages, qid, pid in dataloader_train:
            # print(qid, [q[:-2] for q in qid])
            qid = [int(q[:-2]) for q in qid]
            # print(qid)
            queries = {k: v.to(device) for k, v in queries.items()}
            passages = {k: v.to(device) for k, v in passages.items()}

            q_embs = model(**queries)
            p_embs = model(**passages)

            qid_tensor = torch.tensor(qid)

            loss = loss_fn(q_embs, qid_tensor, ref_emb=p_embs, ref_labels=qid_tensor).mean()

            if len(memory_queue) > 0:
                memory_embs, memory_labels = memory_queue.get_embeddings()
                loss += loss_fn(
                    q_embs, qid_tensor, ref_emb=memory_embs, ref_labels=memory_labels
                ).mean()
            # print(len(memory_queue))
            memory_queue.add_to_memory(p_embs, qid_tensor, batch_size=p_embs.shape[0])

            loss /= gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if step % gradient_accumulation_steps == 0 and (
                step != 0 or gradient_accumulation_steps == 0
            ):
                if cfg.train_args.gradient_clipping is not None:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.train_args.gradient_clipping,
                    )

                optimizer.step()
                optimizer.zero_grad()

                if step < 10_000 and cfg.train_args.verbose:
                    print(f'{step:7}, {running_loss:10.6f}', flush=True)

                if cfg.train_args.log:
                    wandb.log({'loss': running_loss, 'step': step // gradient_accumulation_steps})

                if cfg.train_args.lr_scheduler is not None:
                    scheduler.step()

                running_loss = 0

            run_validation: bool = ((step / gradient_accumulation_steps) % cfg.train_args.steps_between_val) == 0
            run_validation = run_validation and step > gradient_accumulation_steps
            run_validation = run_validation or (step == 0 and cfg.train_args.run_val_check)
            if run_validation:
                val_metrics = validator.validate(model)
                ndcg = val_metrics['ndcg_cut_10']
                mrr = val_metrics['recip_rank']

                if cfg.train_args.log:
                    wandb.log({'ndcg@10': ndcg, 'step': step // gradient_accumulation_steps, 'mrr': mrr})

            if (
                (step / gradient_accumulation_steps) % cfg.train_args.steps_between_save
                == 0
                and (step > gradient_accumulation_steps)
                and cfg.train_args.save_checkpoints
            ):
                save_name = f'step={step // gradient_accumulation_steps}.ckpt'
                full_save_path = os.path.join(save_dir, save_name)
                torch.save(model.state_dict(), full_save_path)
                model.train()
            step += 1


@hydra.main(config_path=None, config_name='config')
def main(cfg: TrainingConfig) -> None:
    run(cfg)


if __name__ == '__main__':
    main()
