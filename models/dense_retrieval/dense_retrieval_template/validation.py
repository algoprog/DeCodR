import torch

import numpy as np

from models import EmbeddingModel
from encoding_indexing_package.encode import Encoder, EncoderArguments
from encoding_indexing_package.index import Indexer
from ir_dataset_package.dataset import CollectionDataset, QueryDataset

from typing import Union, Dict, Any, Tuple, List, Optional
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast, AutoTokenizer, AutoModel


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
            tokenizer=self.tokenizer 
        )

        self.query_dataset = QueryDataset(
            query_path=query_path,
            cache_dir=self.cache_dir,
            tokenizer=self.tokenizer,
        )

    def validate(self, model, args: Optional[EncoderArguments] = None):
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
        encoded_collection, collection_labels = encoder.encode_memory(self.collection_dataset)
        print('Encoding queries')
        encoded_queries, query_labels = encoder.encode_memory(self.query_dataset, collator_kwargs={'id_key': 'qid'})

        print('Done encoding collection', type(encoded_collection), encoded_collection[0].shape, collection_labels[:10])
        print('Done encoding queries', type(encoded_queries), encoded_queries[0].shape, query_labels[:10])

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
        return index.trec_metrics(encoded_queries, query_labels, qrels_path=self.qrels_path)


def load_model(path: str):
    model = EmbeddingModel('distilbert-base-uncased', mlp_hidden_layer_size=1024)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model


def default_processing_fn(outputs, batch):
    return outputs.cpu().numpy()


def bert_processing_fn(outputs, batch):
    return outputs.last_hidden_state[:, 0].cpu().numpy()


def processing_before_indexing_fn(embeddings):
    return np.concatenate(embeddings, axis=0)


def run_validation(model_path: str, tokenizer_name_or_path: str, collection_path, query_path, qrels_path, cache_dir):
    print('Running validation')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    # model = AutoModel.from_pretrained('sentence-transformers/msmarco-bert-co-condensor')
    model = load_model(model_path)

    # collection_path = query_path
    validator = Validator(tokenizer, collection_path, query_path, qrels_path, 
                cache_dir, post_processing_fn=bert_processing_fn, processing_before_indexing_fn=processing_before_indexing_fn)
    validator.validate(model)


def main():
    run_validation(
        model_path='',
        tokenizer_name_or_path='distilbert-base-uncased',
        collection_path='',
        query_path='',
        qrels_path='',
        cache_dir='',
    )


if __name__ == '__main__':
    main()