from optparse import Option
import os
import csv
import hashlib
import random
import datasets
import pickle

from pathlib import Path
from collections import defaultdict
from builtins import FileNotFoundError, TypeError
from typing import Union, Dict, Any, Tuple, List, Optional
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer
from functools import partial


class BaseIrDataset(Dataset):
    def __init__(self, cache_dir: Optional[str]) -> None:
        super().__init__()
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        pass

    def __getitem__(self, index) -> Any:
        pass

    def string_to_int(self, string: str) -> Union[str, int]:
        if type(string) is str and string.isnumeric():
            return int(string)
        else:
            return string

    def _get_file_hash(self, path: str) -> str:
        with open(path, 'rb') as f:
            file_hash = hashlib.md5()
            chunk = f.read(8192)
            while chunk:
                file_hash.update(chunk)
                chunk = f.read(8192)

        return file_hash.hexdigest()

    def _load_or_create_filename_hash(
        self, referenced_path: str, always_return_path: bool = False
    ) -> Dict[Union[str, int], int]:
        if not os.path.isfile(referenced_path):
            raise FileNotFoundError(f'{referenced_path} could not be found')
        filehash = self._get_file_hash(referenced_path)

        if self.cache_dir is None:
            return None

        filepath = os.path.join(self.cache_dir, f'{filehash}.pickle')

        loaded_file = self.load_if_exists(filepath)

        if loaded_file is None or always_return_path:
            return filepath
        else:
            return loaded_file

    def load_if_exists(self, path: str) -> Union[None, Any]:
        file = None
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                file = pickle.load(f)
        return file

    def save_file(self, obj: object, path: str) -> None:
        if path is not None:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)


class CollectionIrDatasetBase(BaseIrDataset):
    collection_keys = ['pid', 'text']

    def __init__(self, collection_path: str, cache_dir: Optional[str]) -> None:
        super().__init__(cache_dir)

        self.collection_path = collection_path
        self.collection = datasets.load_dataset(
            'csv',
            data_files=collection_path,
            delimiter='\t',
            column_names=self.collection_keys,
            cache_dir=self.cache_dir,
            split='train',
        )

        self.collection_lookup = self.load_or_create_collection_lookup()

    def load_or_create_collection_lookup(
        self, force_create_new: bool = False
    ) -> Dict[Union[str, int], int]:
        collection_lookup_path = self._load_or_create_filename_hash(
            self.collection_path
        )

        # In the case that a none string is returned that means the collection lookup was loaded
        # from cache
        if (
            type(collection_lookup_path) is not str
            and collection_lookup_path is not None
            and not force_create_new
        ):
            return collection_lookup_path

        collection_lookup = {
            self.string_to_int(k): v for v, k in enumerate(self.collection['pid'])
        }

        if self.cache_dir is not None and not force_create_new:
            self.save_file(collection_lookup, collection_lookup_path)
        return collection_lookup


class TokenizedMultiNegativeDataset(CollectionIrDatasetBase):
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        collection_path: str,
        query_path: str,
        triplet_ids_path: str,
        cache_dir: Optional[str],
        num_negatives: int = 1,
        num_positives: int = 1,
        query_keys: Optional[List[str]] = None,
        lazy_tokenize: bool = False,
    ) -> None:
        super().__init__(collection_path, cache_dir)
        self.tokenizer = load_tokenizer(tokenizer)
        self.query_path = query_path
        self.triplet_ids_path = triplet_ids_path
        self.num_negatives = num_negatives
        self.num_positives = num_positives
        self.query_keys = query_keys
        self.lazy_tokenize = lazy_tokenize

        if self.query_keys is None:
            self.query_keys = ['qid', 'text']

        self.queries = datasets.load_dataset(
            'csv',
            data_files=self.query_path,
            delimiter='\t',
            column_names=self.query_keys,
            cache_dir=self.cache_dir,
            split='train',
        )

        def tokenize_fn(examples, max_length):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            )

        c_tokenize_fn = partial(tokenize_fn, max_length=128)
        q_tokenize_fn = partial(tokenize_fn, max_length=32)
        # self.collection = self.collection.select(list(i for i in range(20_000)))

        if not self.lazy_tokenize:
            print('Tokenizing collection', flush=True)
            self.collection = self.collection.map(c_tokenize_fn, batched=True)
            print('Tokenizing queries', flush=True)
            self.queries = self.queries.map(q_tokenize_fn, batched=True)

        self.query_lookup = self.load_or_create_query_lookup()
        print('Creating triplet data', flush=True)
        self.training_data = self.load_or_create_triplet_data()
        print('Done creating triplet data', flush=True)

        # small_training_data = []

        # for i in range(2000):
        #     small_training_data.append((int(self.queries[i]['qid']), [i], [i + 1]))

        # print(f'small_training_data length: {len(small_training_data)}')
        # self.training_data = small_training_data

    def __len__(self) -> int:
        return len(self.training_data)

    def _lazy_tokenizer(
        self, input: Dict[str, Any], max_length: int = 128
    ) -> Dict[str, Any]:
        tok_output = self.tokenizer(
            input['text'],
            max_length=max_length,
            truncation=True,
            return_token_type_ids=False,
        )

        # for k, v in input.items():
        #     tok_output[k] = v

        return tok_output

    def __getitem__(self, index) -> Any:
        qid, pos_pids, neg_pids = self.training_data[index]
        query = self.queries[self.query_lookup[qid]]

        pos_start_idx = random.randint(0, len(pos_pids) - self.num_positives)
        neg_start_idx = random.randint(0, len(neg_pids) - self.num_negatives)
        pos_passages = [
            self.collection[self.collection_lookup[pid]]
            for pid in pos_pids[pos_start_idx : pos_start_idx + self.num_positives]
        ]
        neg_passages = [
            self.collection[self.collection_lookup[pid]]
            for pid in neg_pids[neg_start_idx : neg_start_idx + self.num_negatives]
        ]

        if self.lazy_tokenize:
            tok_query = self._lazy_tokenizer(query, max_length=32)
            tok_pos_passages = [
                self._lazy_tokenizer(passage) for passage in pos_passages
            ]
            tok_neg_passages = [
                self._lazy_tokenizer(passage) for passage in neg_passages
            ]
        else:
            tok_query = query
            tok_pos_passages = pos_passages
            tok_neg_passages = neg_passages

        return tok_query, tok_pos_passages, tok_neg_passages

    def load_or_create_triplet_data(self):
        triplet_data_path = self._load_or_create_filename_hash(
            self.triplet_ids_path, always_return_path=False
        )

        if triplet_data_path is not None and type(triplet_data_path) is not str:
            # output_data = []
            # for qid, pos_pids, neg_pids in triplet_data_path:
            #     output_data.append((qid, sorted(list(set(pos_pids))), sorted(list(set(neg_pids)))))
            # triplet_data_path = self._load_or_create_filename_hash(self.triplet_ids_path, always_return_path=True)
            # self.save_file(output_data, triplet_data_path)
            # return output_data
            return triplet_data_path

        print(f'triplet_data_path {triplet_data_path}', flush=True)
        triplet_data = defaultdict(lambda: defaultdict(set))
        i = 0
        with open(self.triplet_ids_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for qid, pos_pid, neg_pid in reader:
                # TODO remove string_to_int here and do it after items are
                qid, pos_pid, neg_pid = (
                    self.string_to_int(qid),
                    self.string_to_int(pos_pid),
                    self.string_to_int(neg_pid),
                )
                triplet_data[qid]['pos_pids'].add(pos_pid)
                triplet_data[qid]['neg_pids'].add(neg_pid)

                # if i > 1_000:
                #     break
                if i % 1_000_000 == 0:
                    print(i)
                i += 1

        output_data = []
        for qid, values in triplet_data.items():
            output_data.append(
                (
                    qid,
                    sorted(list(values['pos_pids'])),
                    sorted(list(values['neg_pids'])),
                )
            )

        output_data = sorted(output_data)

        self.save_file(output_data, triplet_data_path)
        return output_data

    def load_or_create_query_lookup(self) -> Dict[Union[str, int], str]:
        query_lookup = None

        query_lookup_path = self._load_or_create_filename_hash(self.query_path)

        if query_lookup_path is not None and type(query_lookup_path) is not str:
            return query_lookup_path

        query_lookup = {
            self.string_to_int(k): v for v, k in enumerate(self.queries['qid'])
        }
        self.save_file(query_lookup, query_lookup_path)

        return query_lookup


class TokenizedMultiNegativeDatasetTevatron(TokenizedMultiNegativeDataset):
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        collection_path: str,
        query_path: str,
        triplet_ids_path: str,
        cache_dir: Optional[str],
        num_negatives: int = 1,
        query_keys: Optional[List[str]] = None,
        lazy_tokenize: bool = False,
    ) -> None:
        super().__init__(
            tokenizer,
            collection_path,
            query_path,
            triplet_ids_path,
            cache_dir,
            num_negatives,
            num_positives=1,
            query_keys=query_keys,
            lazy_tokenize=lazy_tokenize,
        )

    def __getitem__(self, index) -> Any:
        qry, pos_psg, neg_psg = super().__getitem__(index)
        return [qry], pos_psg + neg_psg

class TokenizedQrelPairDataset(CollectionIrDatasetBase):
    def __init__(
        self,
        tokenizer: Union[str, PreTrainedTokenizer, PreTrainedTokenizerFast],
        collection_path: str,
        query_path: str,
        qrels_path: str,
        cache_dir: Optional[str],
        lazy_tokenize: bool = False,
        query_keys: List[str] = None,
        qrel_keys: List[str] = None,
        query_max_length: int = 32,
        passage_max_length: int = 128,
    ) -> None:
        super().__init__(collection_path, cache_dir)
        self.tokenizer = load_tokenizer(tokenizer)
        self.query_path = query_path
        self.lazy_tokenize = lazy_tokenize
        self.cache_dir = cache_dir
        self.query_keys = query_keys
        self.lazy_tokenize = lazy_tokenize
        self.qrel_keys = qrel_keys
        self.query_max_length = query_max_length
        self.passage_max_length = passage_max_length

        if self.query_keys is None:
            self.query_keys = ['qid', 'text']

        if self.qrel_keys is None:
            self.qrel_keys = ['qid', 'iteration', 'pid', 'relevancy']

        self.queries = datasets.load_dataset(
            'csv',
            data_files=self.query_path,
            delimiter='\t',
            column_names=self.query_keys,
            cache_dir=self.cache_dir,
            split='train',
        )

        def tokenize_fn(examples, max_length):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                max_length=max_length,
                return_token_type_ids=False,
            )

        c_tokenize_fn = partial(tokenize_fn, max_length=passage_max_length)
        q_tokenize_fn = partial(tokenize_fn, max_length=query_max_length)

        if not self.lazy_tokenize:
            print('Tokenizing collection', flush=True)
            self.collection = self.collection.map(c_tokenize_fn, batched=True)
            print('Tokenizing queries', flush=True)
            self.queries = self.queries.map(q_tokenize_fn, batched=True)

        self.query_lookup = self.load_or_create_query_lookup()

        self.qrels = datasets.load_dataset(
            'csv',
            data_files=qrels_path,
            delimiter='\t',
            column_names=self.qrel_keys,
            cache_dir=self.cache_dir,
        )['train']

    def __len__(self) -> int:
        return len(self.qrels)

    def _lazy_tokenizer(
        self, input: Dict[str, Any], max_length: int = 128
    ) -> Dict[str, Any]:
        tok_output = self.tokenizer(
            input['text'],
            max_length=max_length,
            truncation=True,
            return_token_type_ids=False,
        )

        # for k, v in input.items():
        #     tok_output[k] = v

        return tok_output

    def __getitem__(self, index: int) -> Any:
        qrel = self.qrels[index]
        qid, pid = self.string_to_int(qrel['qid']), self.string_to_int(qrel['pid'])
        query = self.queries[self.query_lookup[qid]]
        passage_index = self.collection_lookup[pid]
        passage = self.collection[passage_index]

        if self.lazy_tokenize:
            tok_query = self._lazy_tokenizer(query, max_length=self.query_max_length)
            tok_pos_passages = self._lazy_tokenizer(
                passage, max_length=self.passage_max_length
            )
        else:
            tok_query = query
            tok_pos_passages = passage

        return tok_query, tok_pos_passages, qid, pid

    def load_or_create_query_lookup(self) -> Dict[Union[str, int], str]:
        query_lookup = None

        query_lookup_path = self._load_or_create_filename_hash(self.query_path)

        if query_lookup_path is not None and type(query_lookup_path) is not str:
            return query_lookup_path

        query_lookup = {
            self.string_to_int(k): v for v, k in enumerate(self.queries['qid'])
        }
        self.save_file(query_lookup, query_lookup_path)

        return query_lookup


class CollectionDataset(BaseIrDataset):
    collection_keys = ['pid', 'text']

    def __init__(
        self,
        collection_path: str,
        cache_dir: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        lazy_tokenize: bool = False,
        max_length: int = 512,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.collection_path = collection_path
        self.cache_dir = cache_dir
        self.lazy_tokenize = lazy_tokenize
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.collection = datasets.load_dataset(
            'csv',
            data_files=collection_path,
            delimiter='\t',
            column_names=self.collection_keys,
            cache_dir=self.cache_dir,
        )['train']

        if not self.lazy_tokenize:
            print('Tokenizing collection', flush=True)
            self.collection = self.collection.map(
                lambda examples: self.tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=self.max_length,
                    return_token_type_ids=False,
                ),
                batched=True,
            )

    def __len__(self) -> int:
        return len(self.collection)

    def __getitem__(self, index: int) -> Any:
        item = self.collection[index]

        if self.lazy_tokenize:
            tokenized_item = self.tokenizer(
                item['text'],
                truncation=True,
                max_length=self.max_length,
                return_token_type_ids=False,
            )

            tokenized_item['text'] = item['text']
            tokenized_item['pid'] = item['pid']
        else:
            tokenized_item = item

        return tokenized_item


class QueryDataset(BaseIrDataset):
    query_keys = ['qid', 'text']

    def __init__(
        self,
        query_path: str,
        cache_dir: str,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        lazy_tokenize: bool = False,
        tokenizer_kwargs: Dict[str, str] = {},
        max_length: int = 512,
        max_num_queries: Optional[int] = None,
    ) -> None:
        super().__init__(cache_dir=cache_dir)
        self.query_path = query_path
        self.cache_dir = cache_dir
        self.lazy_tokenize = lazy_tokenize
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs
        self.max_length = max_length
        self.max_num_queries = max_num_queries

        self.queries = datasets.load_dataset(
            'csv',
            data_files=query_path,
            delimiter='\t',
            column_names=self.query_keys,
            cache_dir=self.cache_dir,
        )['train']

        if not self.lazy_tokenize:
            print('Tokenizing queries', flush=True)
            self.queries = self.queries.map(
                lambda examples: self.tokenizer(
                    examples['text'],
                    truncation=True,
                    return_token_type_ids=False,
                    max_length=self.max_length,
                    **self.tokenizer_kwargs,
                ),
                batched=True,
            )

    def __len__(self) -> int:
        if self.max_num_queries is None:
            return len(self.queries)
        else:
            return self.max_num_queries

    def __getitem__(self, index: int) -> Any:
        item = self.queries[index]

        if self.lazy_tokenize:
            tokenized_item = self.tokenizer(
                item['text'],
                truncation=True,
                return_token_type_ids=False,
                max_length=self.max_length,
                **self.tokenizer_kwargs,
            )

            tokenized_item['text'] = item['text']
            tokenized_item['qid'] = item['qid']
        else:
            tokenized_item = item

        return tokenized_item


def load_tokenizer(tokenizer):
    if type(tokenizer) is not str:
        return tokenizer
    else:
        return AutoTokenizer.from_pretrained(tokenizer)

