import os
import torch
import pickle
import torch.nn as nn

from typing import Any, Optional, Union
from contextlib import nullcontext
from typing import Union, Dict, Any, Tuple, List, Optional

from tqdm import tqdm
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

@dataclass
class EncoderArguments:
    batch_size: int = 64
    num_workers: int = 0
    device: str = 'cuda'
    fp16: bool = False
    max_entries_in_memory: Optional[int] = 50_000
    save_dir: str = ''


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __init__(self, *args, **kwargs):            
        self.id_key = kwargs.pop('id_key', 'pid')
        super().__init__(*args, **kwargs)

    def __call__(self, features):
        text_ids = [feat[self.id_key] for feat in features]
        text_features = []
        for feat in features:
            text_features.append({k:v for k,v in feat.items() if k not in [self.id_key, 'text']})
        collated_features = super().__call__(text_features)
        return text_ids, collated_features


class Encoder:
    def __init__(
        self, 
        model: Union[nn.Module, PreTrainedModel, str], 
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast, str], 
        encoder_args: EncoderArguments, 
        post_processing_fn: Optional[Any] = None, 
        save_preprocessing_fn: Optional[Any] = None, 
    ) -> None:
        self.args = encoder_args
        self.post_processing_fn = post_processing_fn
        self.save_preprocessing_fn = save_preprocessing_fn

        if type(tokenizer) is str:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        if type(model) is str:
            self.model = AutoModel.from_pretrained(model)
        else:
            self.model = model

        model = model.to(self.args.device)
        model.eval()

    def save_data(self, encoded, lookup_indices, filename: str) -> None:
        save_path = os.path.join(self.args.save_dir, filename)
        Path(self.args.save_dir).mkdir(parents=True, exist_ok=True)
        print(f'Saving to {save_path}', flush=True)
        with open(save_path, 'wb') as f:
            if self.save_preprocessing_fn is not None:
                pickle.dump(self.save_preprocessing_fn(encoded, lookup_indices), f)
            else:
                pickle.dump((encoded, lookup_indices), f)

    def encode_memory(self, dataset, model_kwargs={}, collator_kwargs: Dict[str, Any] = {}):
        batch_size = self.args.batch_size

        encode_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=EncodeCollator(
                self.tokenizer, return_tensors='pt', **collator_kwargs
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
        )

        encoded = []
        lookup_indices = []
        for batch_idx, (ids, batch) in tqdm(enumerate(encode_loader), total=len(encode_loader)):
            with torch.cuda.amp.autocast() if self.args.fp16 else nullcontext():
                with torch.no_grad():
                    lookup_indices.extend(ids)
                    model_inputs = {k:v.to(self.args.device) for k, v in batch.items()}
                    outputs = self.model(**model_inputs, **model_kwargs)

                if self.post_processing_fn is not None:
                    outputs = self.post_processing_fn(outputs, batch)
                else:
                    outputs = outputs.cpu().numpy()

                if type(outputs) is list:
                    encoded += outputs
                else:
                    encoded.append(outputs)
        return encoded, lookup_indices

    def encode(self, dataset, model_kwargs={}):
        batch_size = self.args.batch_size

        encode_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=EncodeCollator(
                self.tokenizer, return_tensors='pt'
            ),
            shuffle=False,
            drop_last=False,
            num_workers=self.args.num_workers,
        )

        encoded = []
        lookup_indices = []
        max_entries_in_memory = self.args.max_entries_in_memory
        shard_num = 0
        entries = 0
        for batch_idx, (pids, batch) in enumerate(encode_loader):
            entries += batch_size
            if max_entries_in_memory is not None and entries > max_entries_in_memory:
                self.save_data(encoded, lookup_indices, f'{shard_num}.pickle')
                shard_num += 1
                encoded = []
                lookup_indices = []
                entries = 0

            with torch.cuda.amp.autocast() if self.args.fp16 else nullcontext():
                with torch.no_grad():
                    lookup_indices.extend(pids)
                    model_inputs = {k:v.to(self.args.device) for k, v in batch.items()}
                    outputs = self.model(**model_inputs, **model_kwargs)

                if self.post_processing_fn is not None:
                    outputs = self.post_processing_fn(outputs, batch)
                else:
                    outputs = outputs.cpu().numpy()

                if type(outputs) is list:
                    encoded += outputs
                else:
                    encoded.append(outputs)

        self.save_data(encoded, lookup_indices, f'{shard_num}.pickle')
            

        
