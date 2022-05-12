import json
import logging
import os
import re
from collections import defaultdict

from typing import List

import numpy
from flask import Flask, request
from flask_cors import CORS
from gensim.models import Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument
from syntok.tokenizer import Tokenizer
from tqdm import tqdm
from vector_index import VectorIndex

logging.getLogger().setLevel(logging.INFO)


def filter_ngrams(ngrams):
    r = re.compile(r'[A-Za-z ]+$')
    ngrams = [ngram for ngram in ngrams if re.match(r, ngram)]
    filtered_ngrams = []
    r2 = re.compile(r'^(the|a|an|and|in|for|with|from) ')
    r3 = re.compile(r'^(.)* (the|a|an|and|in|for|with|from)$')
    for ngram in ngrams:
        if re.match(r2, ngram):
            continue
        if re.match(r3, ngram):
            continue
        filtered_ngrams.append(ngram)
    return filtered_ngrams


def get_ngrams(tokenizer: Tokenizer, text: str, min_length=1, max_length=3) -> List[str]:
    """
    Gets the word ngrams of a string
    :param tokenizer: tokenizer object
    :param text: the string used to generate ngrams
    :param min_length: the minimum length og the generated ngrams in words
    :param max_length: the maximum length og the generated ngrams in words
    :return: list of ngrams (strings)
    """
    tokens = tokenizer.tokenize(text.lower())
    tokens = [token.value for token in tokens]
    max_length = min(max_length, len(tokens))
    all_ngrams = []
    for n in range(min_length-1, max_length):
        ngrams = [" ".join(ngram) for ngram in zip(*[tokens[i:] for i in range(n+1)])]
        for ngram in ngrams:
            all_ngrams.append(ngram)

    filtered_ngrams = filter_ngrams(all_ngrams)

    return filtered_ngrams


def train(data_path="data", epochs=20):
    tokenizer = Tokenizer()
    logging.info("loading training docs...")

    document_queries = defaultdict(lambda: [])
    with open(f"{data_path}/queries_train.jsonl", encoding="utf8") as f:
        for line in f:
            d = json.loads(line.rstrip("\n"))
            for doc_id in d["docs"]:
                document_queries[doc_id].append(d["query"])

    documents = []
    # Read passages
    with open(f"{data_path}/docs.jsonl", encoding="utf8") as f:
        for i, line in tqdm(enumerate(f)):
            d = json.loads(line.rstrip("\n"))
            name = d["name"]
            description = d["description"] if d["description"] is not None else ""
            readme = d["readme"] if d["readme"] is not None else ""
            tags = [str(d["id"])] + document_queries[d["id"]]
            documents.append(TaggedDocument(
                words=get_ngrams(tokenizer, f"{name} - {description} - {readme}", 1, 3),
                tags=tags
            ))

    logging.info("training...")

    model = Doc2Vec(vector_size=100,
                    dbow_words=1,
                    window=20,
                    negative=8,
                    alpha=0.025,
                    min_alpha=0.00025,
                    min_count=3,
                    seed=42,
                    dm=0,
                    workers=8)
    model.build_vocab(documents)
    model.train(documents,
                total_examples=model.corpus_count,
                epochs=epochs)

    model.save("model.doc2vec")


class Doc2vecRetriever:
    def __init__(self):
        self.model = Doc2Vec.load("model.doc2vec")
        self.tokenizer = Tokenizer()

    def index_documents(self, docs, ids, use_gpu=True):
        self.index = VectorIndex(100)
        self.ids = ids
        for i, doc_id in enumerate(self.ids):
            vector = self.model.dv[doc_id]
            self.index.add(vector)

        self.index.build(use_gpu=use_gpu)

    def search(self, query, topk=100):
        if query in self.model.dv:
            vector = self.model.dv[query]
        else:
            vector = self.model.infer_vector(get_ngrams(self.tokenizer, query.lower()), epochs=150)
        ids, scores = self.index.search([vector], k=topk)
        ids = ids[0]
        scores = scores[0]
        results = [(self.ids[id_], score) for id_, score in zip(ids, scores)]
        return results


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    #train(epochs=10)
    retriever = Doc2vecRetriever()
    from evaluate import evaluate
    evaluate(retriever, split="test")
