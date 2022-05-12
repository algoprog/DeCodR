import logging
import math
import random
import wandb
import torch
import os

os.environ["NCCL_DEBUG"] = "INFO"

from collections import defaultdict
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from reranker import ReRanker, PointwiseRankingDataset, RankingDataset





if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    """
    train(output_path="../weights_ranker_5",
          model_type="distilroberta-base",
          use_snippets=True,
          train_batch_size=64,
          eval_batch_size=64,
          accumulation_steps=1,
          lr=1e-5,
          warmup_steps=1000,
          max_seq_length=512,
          epochs=100,
          eval_steps=1000,
          log_steps=10,
          wandb_log=True,
          train_path="../data/train_full_neg_2.jsonl",
          dev_path="../data/dev_full_neg.jsonl",
          test_path="../data/test_full_neg_4.jsonl",
          use_gpu=True,
          parallel=True,
          fp16=False)
    """

    evaluate()
