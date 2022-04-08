from transformers import DataCollatorWithPadding
from dataclasses import dataclass, field


@dataclass
class TokenizedMultiNegativeCollator(DataCollatorWithPadding):
    valid_keys = ['input_ids', 'attention_mask']

    def clean_keys(self, input):
        if type(input) is list:
            output = []
            for item in input:
                output.append({k: v for k, v in item.items() if k in self.valid_keys})
            return output
        else:
            return {k: v for k, v in input.items() if k in self.valid_keys}

    def __call__(self, features):
        query = [self.clean_keys(feature[0]) for feature in features]

        pos_passages, neg_passages = [], []
        for feature in features:
            pos_passages += self.clean_keys(feature[1])
            neg_passages += self.clean_keys(feature[2])

        collated_query = super().__call__(query)
        collated_pos_passages = super().__call__(pos_passages)
        collated_neg_passages = super().__call__(neg_passages)

        return collated_query, collated_pos_passages, collated_neg_passages


@dataclass
class TokenizedQrelPairCollator(DataCollatorWithPadding):
    valid_keys = ['input_ids', 'attention_mask']

    def clean_keys(self, input):
        if type(input) is list:
            output = []
            for item in input:
                output.append({k: v for k, v in item.items() if k in self.valid_keys})
            return output
        else:
            return {k: v for k, v in input.items() if k in self.valid_keys}

    def __call__(self, features):
        query, passages, qids, pids = [], [], [], []
        for feature in features:
            query.append(self.clean_keys(feature[0]))
            passages.append(self.clean_keys(feature[1]))
            qids.append(feature[2])
            pids.append(feature[3])

        collated_query = super().__call__(query)
        collated_passages = super().__call__(passages)

        return collated_query, collated_passages, qids, pids
