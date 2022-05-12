import json
import random

train = open("data/queries_train.jsonl", "w+")
dev = open("data/queries_dev.jsonl", "w+")
test = open("data/queries_test.jsonl", "w+")

with open("data/queries.jsonl") as f:
    for i, line in enumerate(f):
        d = json.loads(line.rstrip("\n"))
        query = d["query"]
        type_ = d["type"]
        docs = d["docs"]
        random.shuffle(docs)

        train_num = int(len(docs)*0.8)
        dev_num = int((len(docs)-train_num)/2)

        train.write(json.dumps({
            "query": query,
            "qid": i,
            "type": type_,
            "docs": docs[:train_num]
        })+"\n")

        dev.write(json.dumps({
            "query": query,
            "qid": i,
            "type": type_,
            "docs": docs[train_num:train_num+dev_num]
        }) + "\n")

        test.write(json.dumps({
            "query": query,
            "qid": i,
            "type": type_,
            "docs": docs[train_num+dev_num:]
        }) + "\n")
