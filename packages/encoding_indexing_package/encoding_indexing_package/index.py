
import json
import os
import faiss
import torch
import pickle
import psutil

import pytrec_eval

import torch.nn as nn
import numpy as np

from contextlib import nullcontext
from collections import defaultdict
from typing import Any, Optional, Union, List
from builtins import FileNotFoundError, TypeError

from pathlib import Path
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from transformers import (
    DataCollatorWithPadding, 
    PreTrainedTokenizer, 
    PreTrainedTokenizerFast,
    AutoTokenizer,
    PreTrainedModel,
    AutoModel,
)


class Indexer:
    '''
        processing_before_indexing_fn: 
            Called on data that does not have one embedding/representation per passage id
            Allows what data to be added to index to be reduced or modified

            Input: List[np.array (shape M, D)]
            Outputs: List[np.array (shape, K, D)] where K can be equal to M or not be equal to M
    '''
    def __init__(
        self, 
        index_type: str = '', 
        data_dim: int = 64,
        normalize_embeddings: bool = False,
        processing_before_indexing_fn: Optional[Any] = None,
        process_on_load: bool = True,
    ) -> None:
        self.processing_before_indexing_fn = processing_before_indexing_fn
        self.labels = []

        # self.index = faiss.index_factory()
        self.process_on_load = process_on_load
        self.data_dim = data_dim
        self.normalize_embeddings = normalize_embeddings
        self.index = faiss.IndexFlatIP(self.data_dim) 
        self.embeddings = None

    def add_data(
        self, 
        embeddings: Optional[Union[List[List[float]], List[List[List[float]]], List[np.array], List[List[np.array]], np.array]] = None,
        pids: Optional[Union[List[int], np.array]] = None,
        data_dir: Optional[str] = None,
    ):
        if data_dir is None and embeddings is None and pids is None:
            raise TypeError('Either data_dir or embeddings and pids need to be passed to the add_data method')
        elif data_dir is None and embeddings is None and pids is not None:
            raise TypeError('Values must be passed for both embeddings and pids in add_data but embeddings are None')
        elif data_dir is None and embeddings is not None and pids is None:
            raise TypeError('Values must be passed for both embeddings and pids in add_data but pids are None')
        elif data_dir is not None and embeddings is not None and pids is not None:
            raise TypeError('add_data was passed values for embeddings, pids, and data_dir, but only data_dir or embeddings and pid can be passed')

        if data_dir is not None:
            embeddings, pids = self._load_embeddings(data_dir)
            print('Done loading embeddings')
            # with open('/scratch/jkillingback_umass_edu/ALL/0.pickle', 'wb') as f:
            #     pickle.dump((embeddings, pids), f)

        

        if self.processing_before_indexing_fn is not None and not self.process_on_load:
            embeddings = self.processing_before_indexing_fn(embeddings)
            print('Done processing embeddings', embeddings.shape, embeddings[0].shape)

        if len(embeddings[0].shape) == 2:
            num_passages = len(embeddings)
            sum_num_embs = 0
            for idx, emb in enumerate(embeddings):
                num_embs = emb.shape[0]
                sum_num_embs += num_embs
                self.labels += [pids[idx]] * num_embs
            print(f'Sum of num embs {sum_num_embs} total num passages {num_passages} avg num embs per passage {sum_num_embs / num_passages}')
        else:
            self.labels = pids

        print('Done getting labels')
        self.embeddings = embeddings


    def index_embeddings(self, processing_fn = None):
        if self.embeddings is None:
            raise TypeError('Can not add self.embeddings to index as they are none')

        if processing_fn is not None:
            embeddings = processing_fn(self.embeddings)
        else:
            embeddings = self.embeddings


        print(f'Embeddings type {type(embeddings)}, shape {embeddings.shape}')
        if type(embeddings) is not np.ndarray:
            processed_embs = np.concatenate(embeddings, axis=0)
        else:
            processed_embs = embeddings

        print('Done processing embeddings', processed_embs.shape)
        print('Adding embeddings to index')
        if self.normalize_embeddings:
            faiss.normalize_L2(processed_embs)

        self.index.add(processed_embs)
        print('Done adding embeddings to index')

    def trec_metrics(self, embeddings, labels, qrels_path: str, k: int = 1000, return_search: bool = False):
        run = self._search(embeddings, labels, k)
        run = json.loads(json.dumps(run))

        with open(qrels_path, 'r') as f:
            qrel = pytrec_eval.parse_qrel(f)

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, pytrec_eval.supported_measures)
        results = evaluator.evaluate(run)

        query_measures = [value for value in results.values()][0]
        metrics = {}
        for measure in sorted(query_measures.keys()):
            aggregated_metric = pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure] for query_measures in results.values()]
            )
            metrics[measure] = aggregated_metric

        if return_search:
            return (metrics, run)
        else:
            return metrics
            

    def _search(self, embeddings, labels, k: int = 10):
        if self.normalize_embeddings:
            faiss.normalize_L2(embeddings)

        D, I = self.index.search(embeddings, k=k)
        return self._process_search(D, I, labels)
    
    '''
        Returns data in the form
        {
            qid: {
                pid0: score,
                pid1, score,
            }

        }
    '''
    def _process_search(self, dist_mat, indices_mat, query_labels):
        output = defaultdict(lambda: defaultdict(int))
        for i in range(len(query_labels)):
            q_label = query_labels[i]
            for j, idx in enumerate(indices_mat[i]):
                pid = self.labels[idx]
                output[q_label][pid] += float(dist_mat[i][j])
        return output

    def _standardize_data(self, embeddings, pids):
        pass

    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def _load_embeddings(self, data_dir):
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f'{data_dir} could not be found')

        ordered_files = sorted([(int(file.split('.')[0]), file) for file in os.listdir(data_dir)])
        encoded, lookups = [], []
        for i, (_, file) in enumerate(ordered_files):
            file_path = os.path.join(data_dir, file)
            print(f'Loading {file_path}')
            print(f'Memory used {psutil.virtual_memory().percent}')
            encoded_shard, lookups_shard = self._load_pickle(file_path)

            if self.processing_before_indexing_fn is not None and self.process_on_load:
                encoded_shard = self.processing_before_indexing_fn(encoded_shard)

            if type(encoded_shard) is list:
                encoded += encoded_shard
            elif type(encoded_shard) is np.array:
                encoded.append(encoded_shard)

            if type(lookups_shard) is list:
                lookups += lookups_shard
            elif type(lookups_shard) is np.array:
                lookups.append(lookups_shard)

        return encoded, lookups
        
