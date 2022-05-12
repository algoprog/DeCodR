from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple, Any, Dict, Union


@dataclass
class DataArguments:
    batch_size: int = 8
    val_batch_size: int = 32
    num_workers: int = 0
    cache_dir: str = '/work/jkillingback_umass_edu/cache/'
    collection_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/train/collection.tsv'
    query_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/train/queries.tsv'
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/train/qrels.tsv'
    val_collection_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/val/collection.tsv'
    val_query_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/val/queries.tsv'
    val_qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/val/qrels.tsv'
    lazy_tokenize: bool = True
    query_max_length: int = 512
    passage_max_length: int = 512


@dataclass
class ModelArguments:
    tokenizer_name_or_path: str = 'Luyu/co-condenser-marco'
    model_name_or_path: str = 'Luyu/co-condenser-marco'
    embedding_dim: int = 768
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
    version: str = '5'
    save_dir: str = '/work/jkillingback_umass_edu/checkpoints'
    notes: str = ''
    tags: List[str] = field(default_factory=lambda: [])
    steps_between_save: int = 10_000
    steps_between_val: int = 5_000
    save_checkpoints: bool = True
    log: bool = True
    optimizer: Optimizer = Optimizer(lr=2e-5, weight_decay=0.0)
    lr_scheduler: Optional[LrScheduler] = LinearScheduler(num_warmup_steps=0)
    verbose: bool = True
    max_epochs: int = 100
    gradient_accumulation_steps: int = 8
    gradient_clipping: Optional[float] = 1
    run_val_check: bool = True


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

