import wandb

from transformers import (
    DebertaV2ForMaskedLM,
    DebertaV2Config,
    DebertaV2Tokenizer,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from torch.utils.checkpoint import checkpoint
from dataclasses import dataclass
from datasets import Dataset, load_dataset


@dataclass
class TrainingConfig:
    pretrained_name_or_path: str = 'microsoft/deberta-v3-base'
    pretrained_tokenizer_name_or_path: str = '/work/jkillingback_umass_edu/pretrain-code-deberta/tokenizer/'
    max_position_embeddings: int = 2048
    truncation_length: int = 512
    output_dir: str = '/work/jkillingback_umass_edu/checkpoints/pretrain-code-deberta'
    question_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/large_questions_clean.csv'
    cache_dir: str = '/work/jkillingback_umass_edu/cache'
    log: bool = False


def train(cfg: TrainingConfig):
    print(cfg)
    if cfg.log:
        wandb.init(
            project='pretrain-code-deberta',
            name='pretrain-3',
        )

    tokenizer = DebertaV2Tokenizer.from_pretrained(
        cfg.pretrained_tokenizer_name_or_path
    )

    model_config = DebertaV2Config.from_pretrained(cfg.pretrained_name_or_path)
    model_config.max_position_embeddings = cfg.max_position_embeddings

    model = DebertaV2ForMaskedLM.from_pretrained(
        cfg.pretrained_name_or_path, config=model_config
    )

    model.resize_token_embeddings(len(tokenizer.get_vocab()))

    dataset = load_dataset(
        'csv',
        data_files=cfg.question_path,
        column_names=[
            'id',
            'title',
            'tags',
            'body',
            'acceptedAnswerId',
            'score',
            'views',
        ],
        cache_dir=cfg.cache_dir,
        split='train',
    )

    def tokenize_fn(examples):
        return tokenizer(
            examples['body'],
            truncation=True,
            max_length=cfg.truncation_length,
            return_token_type_ids=False,
            return_special_tokens_mask=True,
        )

    dataset = dataset.map(
        tokenize_fn, batched=True, remove_columns=dataset.column_names
    )
    dataset = dataset.train_test_split(test_size=5_000, seed=1234)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm_probability=0.15
    )

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        evaluation_strategy='steps',
        learning_rate=2e-5,
        weight_decay=0.0,
        per_device_eval_batch_size=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        logging_steps=10,
        fp16=True,
        gradient_checkpointing=False,
        disable_tqdm=True,
        eval_steps=5_000,
        save_steps=10_000,
        group_by_length=True,
        report_to=('wandb' if cfg.log else 'none'),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=data_collator,
    )

    trainer.train()


def main():
    cfg = TrainingConfig()
    train(cfg)


if __name__ == '__main__':
    main()
