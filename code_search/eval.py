import csv
import hydra
import json
import torch
import pytrec_eval

import numpy as np

from encoding_indexing_package.encode import Encoder, EncoderArguments
from encoding_indexing_package.index import Indexer
from ir_dataset_package.dataset import CollectionDataset, QueryDataset

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Union, Dict, Any, Tuple, List, Optional
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModel,
)


@dataclass
class EvalDenseRetrievalConfig:
    query_tsv_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/queries_small_titles.tsv'
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels_small_titles.tsv'
    cache_dir: str = '/work/jkillingback_umass_edu/cache'
    tokenizer_name_or_path: str = 'Luyu/co-condenser-marco'
    encoded_collection_dir: str = '/work/jkillingback_umass_edu/code_search_results/20-000-co-condenser'
    save_run_path: str = '/work/jkillingback_umass_edu/code_search_runs/20_000_duplicate_test_questions.txt'
    # model_name_or_path: str = 'Luyu/co-condenser-marco'
    model_name_or_path: str = '/work/jkillingback_umass_edu/checkpoints/tevatron/checkpoint-20000'
    query_max_length: int = 512
    batch_size: int = 128


@dataclass
class EvalSmallTitlesConfig(EvalDenseRetrievalConfig):
    query_tsv_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/queries_small_titles.tsv'
    qrels_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels_small_titles.tsv'

@dataclass
class EvalSmallQuestionsConfig(EvalDenseRetrievalConfig):
    query_tsv_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/queries_small_questions.tsv'
    qrels_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels_small_questions.tsv'


@dataclass
class EvalDuplicateQuestionsConfig(EvalDenseRetrievalConfig):
    query_tsv_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/queries_q.tsv'
    qrels_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/qrels_q.tsv'

@dataclass
class EvalDuplicateTitlesConfig(EvalDenseRetrievalConfig):
    query_tsv_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/queries_t.tsv'
    qrels_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/qrels_t.tsv'



cs = ConfigStore.instance()
cs.store(name='config', node=EvalDuplicateQuestionsConfig)


def encode_queries(encoder, query_dataset):
    return encoder.encode_memory(
        query_dataset, collator_kwargs={'id_key': 'qid'}
    )


def index_and_search(cfg: EvalDenseRetrievalConfig, encoded_queries, query_labels):
    index = Indexer(
        data_dim=encoded_queries[0].shape[-1],
        processing_before_indexing_fn=processing_before_indexing_fn,
        process_on_load=False,
        normalize_embeddings=False,
    )

    encoded_queries = processing_before_indexing_fn(encoded_queries)
    index.add_data(data_dir=cfg.encoded_collection_dir)
    index.index_embeddings()
    return index.trec_metrics(
        encoded_queries,
        query_labels,
        qrels_path=cfg.qrels_path,
        return_search=False,
    )


def write_run(search, output_path):
    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for qid in search.keys():
            # print(qid)
            for i, (pid, score) in enumerate(search[qid].items()):
                # print(pid)
                writer.writerow([qid, 'Q0', pid, i + 1, score, 'run1'])



def index_and_see_results(cfg: EvalDenseRetrievalConfig, encoded_queries, query_labels):
    index = Indexer(
        data_dim=encoded_queries[0].shape[-1],
        processing_before_indexing_fn=processing_before_indexing_fn,
        process_on_load=False,
        normalize_embeddings=False,
    )

    encoded_queries = processing_before_indexing_fn(encoded_queries)
    index.add_data(data_dir=cfg.encoded_collection_dir)
    index.index_embeddings()
    metrics, search = index.trec_metrics(
        encoded_queries,
        query_labels,
        qrels_path=cfg.qrels_path,
        return_search=True,
    )

    write_run(search, cfg.save_run_path)
    return metrics


def bert_processing_fn(outputs, batch):
    return outputs.last_hidden_state[:, 0].cpu().numpy()


def processing_before_indexing_fn(embeddings):
    return np.concatenate(embeddings, axis=0)


@hydra.main(config_path=None, config_name='config')
def main(cfg: EvalDenseRetrievalConfig) -> None:
    print(cfg)

    model = AutoModel.from_pretrained(cfg.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path)

    args = EncoderArguments(
        batch_size=cfg.batch_size,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    encoder = Encoder(
        model,
        tokenizer=tokenizer,
        encoder_args=args,
        post_processing_fn=bert_processing_fn,
        save_preprocessing_fn=None,
    )

    query_dataset = QueryDataset(
        query_path=cfg.query_tsv_path,
        cache_dir=cfg.cache_dir,
        tokenizer=tokenizer,
        max_length=cfg.query_max_length,
        lazy_tokenize=True,
        max_num_queries=None,
    )

    print('Encoding queries')
    encoded_queries, query_labels = encode_queries(encoder, query_dataset)
    print('Done encoding queries')
    # metrics = index_and_search(cfg, encoded_queries, query_labels)
    metrics = index_and_see_results(cfg, encoded_queries, query_labels)
    
    # print(metrics)
    print(json.dumps(metrics, indent=4))


if __name__ == '__main__':
    main()
