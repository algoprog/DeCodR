import csv
import json
import torch
import pytrec_eval

import numpy as np

from encoding_indexing_package.encode import Encoder, EncoderArguments
from encoding_indexing_package.index import Indexer
from ir_dataset_package.dataset import CollectionDataset, QueryDataset

from typing import Union, Dict, Any, Tuple, List, Optional
from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    AutoTokenizer,
    AutoModel,
)

import csv
import json
import hydra

from dataclasses import dataclass, field
from typing import Dict, Union
from hydra.core.config_store import ConfigStore


@dataclass
class EncodeCollectionConfig:
    collection_tsv_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/train/collection.tsv'
    encode_output_dir: str = '/work/jkillingback_umass_edu/code_search_results/30-000-co-condenser'
    cache_dir: str = '/work/jkillingback_umass_edu/cache'
    tokenizer_name_or_path: str = 'Luyu/co-condenser-marco'
    model_name_or_path: str = '/work/jkillingback_umass_edu/checkpoints/tevatron/checkpoint-30000'
    collection_max_length: int = 512
    batch_size: int = 128
    max_entries_in_memory: int = 1_100_000
    

cs = ConfigStore.instance()
cs.store(name='config', node=EncodeCollectionConfig)

def encode_collection(cfg: EncodeCollectionConfig, model, tokenizer, post_processing_fn):
    collection_dataset = CollectionDataset(
        collection_path=cfg.collection_tsv_path,
        cache_dir=cfg.cache_dir,
        tokenizer=tokenizer,
        max_length=cfg.collection_max_length,
    )

    args = EncoderArguments(
        batch_size=cfg.batch_size,
        save_dir=cfg.encode_output_dir,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        max_entries_in_memory=cfg.max_entries_in_memory
    )

    encoder = Encoder(
        model,
        tokenizer=tokenizer,
        encoder_args=args,
        post_processing_fn=post_processing_fn,
        save_preprocessing_fn=None,
    )

    encoder.encode(collection_dataset)



def bert_processing_fn(outputs, batch):
    return outputs.last_hidden_state[:, 0].cpu().numpy()


def processing_before_indexing_fn(embeddings):
    return np.concatenate(embeddings, axis=0)


@hydra.main(config_path=None, config_name='config')
def main(cfg: EncodeCollectionConfig) -> None:
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path)
    model = AutoModel.from_pretrained(cfg.model_name_or_path)

    encode_collection(cfg, model, tokenizer, bert_processing_fn)


if __name__ == '__main__':
    main()