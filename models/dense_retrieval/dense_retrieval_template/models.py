import torch.nn as nn

from transformers import AutoModel


class EmbeddingModel(nn.Module):
    def __init__(self, model_name_or_path: str,) -> None:
        super().__init__()
        self.muppet = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask):
        muppet_out = self.muppet(input_ids=input_ids, attention_mask=attention_mask)
        cls = muppet_out.last_hidden_state[:, 0]

        return cls

