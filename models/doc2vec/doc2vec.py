import json
import logging
import os
import re

from typing import List

from flask import Flask, request
from flask_cors import CORS
from gensim.models import Doc2Vec
from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.doc2vec import TaggedDocument
from annoy import AnnoyIndex
from syntok.tokenizer import Tokenizer

logging.getLogger().setLevel(logging.INFO)


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

    return filtered_ngrams


def load_dataset():
    logging.info("loading dataset...")
    docs = {}
    
    # TODO

    return docs


def load_train_dataset():
    docs = load_dataset()
    tokenizer = Tokenizer()
    logging.info("loading training docs...")

    train_docs = []
    for doc_id, info in docs.items():
        train_docs.append(TaggedDocument(
            words=get_ngrams(tokenizer, info["title"], 1, 3),
            tags=[doc_id]
        ))
    return train_docs


def train(epochs=20):
    documents = load_train_dataset()
    model = Doc2Vec(vector_size=100,
                    dbow_words=1,
                    window=100,
                    negative=20,
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


class Doc2vecSearch:
    def __init__(self):
        self.docs = load_dataset()
        self.doc_ids = list(self.docs.keys())

        self.model = Doc2Vec.load("model.doc2vec")
        self.tokenizer = Tokenizer()
        self.load_indexes()

        self.app = Flask(__name__)
        CORS(self.app)

    def load_indexes(self):
        self.doc_index = AnnoyIndex(100, metric="angular")
        if os.path.exists("doc_index.ann"):
            logging.info("loading cached doc index...")
            self.doc_index.load("doc_index.ann")
        else:
            logging.info("building doc index...")
            for i, doc_id in enumerate(self.recipe_ids):
                vector = self.model.dv[doc_id]
                self.doc_index.add_item(i, vector)
            self.doc_index.build(100)
            self.doc_index.save("doc_index.ann")

        self.term_index = AnnoyIndex(100, metric="angular")
        if os.path.exists("term_index.ann"):
            logging.info("loading cached term index...")
            self.term_index.load("term_index.ann")
        else:
            logging.info("building term index...")
            for i, key in enumerate(self.model.wv.key_to_index):
                vector = self.model.wv[key]
                self.term_index.add_item(i, vector)
            self.term_index.build(100)
            self.term_index.save("term_index.ann")

        logging.info("loaded indexes")

    def doc_search(self, query):
        vector = self.model.infer_vector(get_ngrams(self.tokenizer, query.lower()), epochs=150)
        results = []
        ids, distances = self.doc_index.get_nns_by_vector(vector, 100, search_k=-1, include_distances=True)
        total_results = len(ids)
        for i in range(total_results):
            similarity = 1 - 0.5 * pow(distances[i], 2)
            results.append({
                "score": similarity,
                "info": self.docs[self.doc_ids[ids[i]]]
            })
        return results

    def term_search(self, query):
        if query.lower() in self.model.wv:
            vector = self.model.wv[query]
        else:
            vector = self.model.infer_vector(get_ngrams(self.tokenizer, query.lower()), epochs=150)
        results = []
        ids, distances = self.term_index.get_nns_by_vector(vector, 100, search_k=-1, include_distances=True)
        total_results = len(ids)
        for i in range(total_results):
            similarity = 1 - 0.5 * pow(distances[i], 2)
            results.append({
                "score": similarity,
                "term": self.model.wv.index_to_key[ids[i]]
            })
        return results

    def serve(self, host='0.0.0.0', port=80):
        @self.app.route('/search', methods=['POST', 'GET'])
        def recipe_search():
            query = request.args.get('q')
            return json.dumps(self.doc_search(query), indent=4)

        @self.app.route('/term_search', methods=['POST', 'GET'])
        def term_search():
            query = request.args.get('q')
            return json.dumps(self.term_search(query), indent=4)

        self.app.run(host=host, port=port)


if __name__ == "__main__":
    #train(epochs=20)
    s = Doc2vecSearch()
    s.serve(port=1234)
