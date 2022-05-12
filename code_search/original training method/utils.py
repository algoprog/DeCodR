import torch
import random

from typing import Optional

import numpy as np

class EmbeddingMemoryQueue:
    def __init__(self, embedding_size, memory_size=1024):
        super().__init__()
        self.embedding_size = embedding_size
        self.memory_size = memory_size
        self.reset_queue()

    def __len__(self):
        if self.has_been_filled:
            return self.memory_size
        else:
            return self.queue_idx

    def add_to_memory(self, embeddings, labels, batch_size):
        self.curr_batch_idx = (
            torch.arange(
                self.queue_idx, self.queue_idx + batch_size, device=labels.device
            )
            % self.memory_size
        )
        self.embedding_memory[self.curr_batch_idx] = embeddings.detach().cpu()
        self.label_memory[self.curr_batch_idx] = labels.detach().cpu()
        prev_queue_idx = self.queue_idx
        self.queue_idx = (self.queue_idx + batch_size) % self.memory_size
        if (not self.has_been_filled) and (self.queue_idx <= prev_queue_idx):
            self.has_been_filled = True

    def get_embeddings(self, device='cuda'):
        if not self.has_been_filled:
            E_mem = self.embedding_memory[: self.queue_idx]
            L_mem = self.label_memory[: self.queue_idx]
        else:
            E_mem = self.embedding_memory
            L_mem = self.label_memory

        return E_mem.to(device=device), L_mem.to(device=device)

    def reset_queue(self):
        self.embedding_memory = torch.zeros(self.memory_size, self.embedding_size)
        self.label_memory = torch.zeros(self.memory_size).long()
        self.has_been_filled = False
        self.queue_idx = 0


def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if not (min_seed_value <= seed <= max_seed_value):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed

def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)


# def split_text_distance_mat(query_embs, passage_embs, qids, pids):
#     '''
#         query_embs shape [# query chunks, 768]
#         passage_embs shape [# passage chunks, 768]

#         passage mask needs to be in form

#         Where the number of 1s in a column correspond to the number of passage
#         chunks
#         [
#         [1, 0]
#         [1, 0]
#         [0, 1]
#         [0, 1]
#         ]

#         query mask needs to be in form

#     '''
#     distance_mat = torch.mm(query_embs, passage_embs.T)

#     passage_mask = 
