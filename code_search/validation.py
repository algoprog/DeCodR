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







class Validator:
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        collection_path: str,
        query_path: str,
        qrels_path: str,
        cache_dir: str,
        post_processing_fn: Optional[Any] = None,
        processing_before_indexing_fn: Optional[Any] = None,
        batch_size: int = 128,
        collection_max_length: int = 512,
        query_max_length: int = 512,
    ) -> None:
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.qrels_path = qrels_path
        self.post_processing_fn = post_processing_fn
        self.processing_before_indexing_fn = processing_before_indexing_fn

        self.collection_dataset = CollectionDataset(
            collection_path=collection_path,
            cache_dir=self.cache_dir,
            tokenizer=self.tokenizer,
            max_length=collection_max_length,
        )

        self.query_dataset = QueryDataset(
            query_path=query_path,
            cache_dir=self.cache_dir,
            tokenizer=self.tokenizer,
            max_length=query_max_length,
        )

    def validate(
        self, model, args: Optional[EncoderArguments] = None, return_search: bool = False
    ):
        if args is None:
            args = EncoderArguments(
                batch_size=self.batch_size,
                save_dir='./validation_outputs',
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )

        encoder = Encoder(
            model,
            tokenizer=self.tokenizer,
            encoder_args=args,
            post_processing_fn=self.post_processing_fn,
            save_preprocessing_fn=None,
        )

        print('Encoding collection')
        encoded_collection, collection_labels = encoder.encode_memory(
            self.collection_dataset
        )
        print('Encoding queries')
        encoded_queries, query_labels = encoder.encode_memory(
            self.query_dataset, collator_kwargs={'id_key': 'qid'}
        )

        print(
            'Done encoding collection',
            type(encoded_collection),
            encoded_collection[0].shape,
            collection_labels[:10],
        )
        print(
            'Done encoding queries',
            type(encoded_queries),
            encoded_queries[0].shape,
            query_labels[:10],
        )

        print('Collection shape', encoded_collection[0].shape)
        index = Indexer(
            data_dim=encoded_collection[0].shape[-1],
            processing_before_indexing_fn=self.processing_before_indexing_fn,
            process_on_load=False,
            normalize_embeddings=False,
        )
        encoded_queries = self.processing_before_indexing_fn(encoded_queries)
        index.add_data(embeddings=encoded_collection, pids=collection_labels)
        index.index_embeddings()
        return index.trec_metrics(
            encoded_queries,
            query_labels,
            qrels_path=self.qrels_path,
            return_search=return_search,
        )


def load_model(path: str):
    model = EmbeddingModel('microsoft/deberta-v3-xsmall')
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    state_dict = {k.replace('module.', ''):v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def display_search_results(collection_path, query_path, qrels_path, search_results_path):
    collection = load_query_or_collection(collection_path)
    queries = load_query_or_collection(query_path)

    with open(qrels_path) as f:
        qrel = pytrec_eval.parse_qrel(f)

    with open(search_results_path) as f:
        search_results = json.load(f)

    for qid in [qid_ for qid_, text in queries.items() if (len(text) < 500 and 'q' in qid_)]:
        p_pids = qrel[qid]
        print(qid, queries[qid])
        print('+' * 50)
        valid_pids = list(p_pids.keys())
        if len(valid_pids) > 0:
            print(collection[valid_pids[0]])
        
        for pid in list(search_results[qid].keys())[:5]:
            print('-' * 50)
            pid_string = pid if pid not in p_pids else f'*{pid}*'
            print(pid_string, collection[pid], end='\n\n')
            
        print('\n\n\n')
        print('=' * 50)


def load_query_or_collection(path):
    values = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for id, text in reader:
            values[id] = text
    return values


def default_processing_fn(outputs, batch):
    return outputs.cpu().numpy()


def bert_processing_fn(outputs, batch):
    return outputs.last_hidden_state[:, 0].cpu().numpy()


def processing_before_indexing_fn(embeddings):
    return np.concatenate(embeddings, axis=0)


def run_validation(
    model_path: str,
    tokenizer_name_or_path: str,
    collection_path,
    query_path,
    qrels_path,
    cache_dir,
):
    print('Running validation')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    model = AutoModel.from_pretrained(model_path)
    # model = load_model(model_path)

    # collection_path = query_path
    validator = Validator(
        tokenizer,
        collection_path,
        query_path,
        qrels_path,
        cache_dir,
        post_processing_fn=bert_processing_fn,
        processing_before_indexing_fn=processing_before_indexing_fn,
        batch_size=256,
    )
    return validator.validate(model, return_search=False)


def main():
    metrics = run_validation(
        model_path='/work/jkillingback_umass_edu/checkpoints/tevatron/checkpoint-10000',
        tokenizer_name_or_path='Luyu/co-condenser-marco',
        collection_path='/work/jkillingback_umass_edu/data/stack-overflow-data/val/collection.tsv',
        query_path='/work/jkillingback_umass_edu/data/stack-overflow-data/val/queries.tsv',
        qrels_path='/work/jkillingback_umass_edu/data/stack-overflow-data/val/qrels.tsv',
        cache_dir='/work/jkillingback_umass_edu/cache/',
    )

    # display_search_results(
    #     '/work/jkillingback_umass_edu/data/stack-overflow-data/val/collection.tsv',
    #     '/work/jkillingback_umass_edu/data/stack-overflow-data/val/queries.tsv',
    #     '/work/jkillingback_umass_edu/data/stack-overflow-data/val/qrels.tsv',
    #     'validation_run_small_model.json',
    # )

    # with open('validation_run_small_model.json', 'w') as f:
    #     json.dump(search, f)

    print(metrics)
    print(json.dumps(metrics, indent=4))
    # print(search)


if __name__ == '__main__':
    main()
