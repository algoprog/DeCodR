from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Any, Dict, Union


@dataclass
class DataArguments:
    batch_size: int = 16
    val_batch_size: int = 64
    num_workers: int = 0
    cache_dir: str = '/mnt/nfs/scratch1/jkillingback/cache/'
    collection_path: str = '/mnt/nfs/scratch1/jkillingback/ANCE_DATA/data/raw_data/collection.tsv'
    query_path: str = '/mnt/nfs/scratch1/jkillingback/ANCE_DATA/data/raw_data/queries.train.tsv'
    qrels_path: str = '/mnt/nfs/scratch1/jkillingback/ANCE_DATA/data/raw_data/qrels.train.tsv'
    val_collection_path: str = '/home/jkillingback/validation-msmarco/data/2_000_k=100/passages.tsv'
    val_query_path: str = '/home/jkillingback/validation-msmarco/data/2_000_k=100/queries.tsv'
    val_qrels_path: str = '/home/jkillingback/validation-msmarco/data/2_000_k=100/qrels.tsv'
    lazy_tokenize: bool = True
    query_max_length: int = 32
    passage_max_length: int = 128


@dataclass
class ModelArguments:
    tokenizer_name_or_path: str = 'distilbert-base-uncased'
    model_name_or_path: str = 'distilbert-base-uncased'
    load_checkpoint_path: Optional[str] = None


@dataclass
class Optimizer:
    _target_: str = 'transformers.AdamW'
    lr: float = 5e-5
    weight_decay: float = 0.0


@dataclass
class SGD(Optimizer):
    _target_: str = 'torch.optim.SGD'
    lr: float = 5e-5
    weight_decay: float = 0.0
    momentum: float = 0.9 


@dataclass
class LrScheduler:
    _target_: str = ''


@dataclass
class LinearScheduler(LrScheduler):
    _target_: str = 'transformers.get_linear_schedule_with_warmup'
    num_warmup_steps: int = 30_000
    num_training_steps: int = 500_000


@dataclass
class TrainingArguments:
    name_prefix: str = 'DenseRetrieval'
    version: str = '1'
    save_dir: str = './checkpoints/'
    notes: str = ''
    tags: List[str] = field(default_factory=lambda: [])
    steps_between_save: int = 10_000
    steps_between_val: int = 10_000
    save_checkpoints: bool = False
    log: bool = False
    optimizer: Optimizer = Optimizer(lr=5e-5, weight_decay=0.0)
    lr_scheduler: Optional[LrScheduler] = None
    verbose: bool = True
    max_epochs: int = 100
    gradient_accumulation_steps: int = 4
    gradient_clipping: Optional[float] = 1


@dataclass
class TrainingConfig:
    data_args: Optional[DataArguments] = DataArguments()
    model_args: Optional[ModelArguments] = ModelArguments()
    train_args: Optional[TrainingArguments] = TrainingArguments()
    hydra: Dict[str, Any] = field(
        default_factory=lambda: {
            'run': {'dir': './outputs/${train_args.name_prefix}/${train_args.version}'}
        }
    )

