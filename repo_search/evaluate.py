import json
import math

import pytrec_eval

from collections import defaultdict
from tqdm import tqdm

from dense_retrieval import DenseEncoder
from doc2vec_retrieval import Doc2vecRetriever
from reranker import ReRanker
from sparse_retrieval import SparseRetriever


def evaluate(retriever, data_path="data", split="dev", rerank=False):
    if rerank:
        reranker = ReRanker(model_path="weights_ranker",
                            model_type="distilroberta-base",
                            use_gpu=True,
                            parallel=True,
                            debug=False,
                            max_seq_length=300)
    else:
        reranker = None

    corpus = {}  # Our whole corpus, pid => passage
    corpus_names = {}
    # Read passages
    with open(f"{data_path}/docs.jsonl", encoding="utf8") as f:
        for line in f:
            d = json.loads(line.rstrip("\n"))
            name = d["name"]
            description = d["description"] if d["description"] is not None else ""
            readme = d["readme"] if d["readme"] is not None else ""
            corpus[d["id"]] = " </s> ".join([name, description, readme])
            corpus_names[d["id"]] = d["name"]

    test_corpus = {}
    test_queries = {}  # Our dev queries. qid => query
    qrel = defaultdict(lambda: {})  # Mapping qid => set with relevant pids
    test_doc_ids = set()
    repo_names = set()
    with open(f"{data_path}/qrels_{split}.jsonl", encoding="utf8") as f:
        for line in f:
            d = json.loads(line.rstrip("\n"))
            test_queries[d["qid"]] = d["query"]
            for doc_id in d["docs"]:
                if corpus_names[doc_id] not in repo_names:
                    test_doc_ids.add(doc_id)
                    qrel[str(d["qid"])][str(doc_id)] = 1
                    repo_names.add(corpus_names[doc_id])

    test_doc_ids = list(test_doc_ids)
    for doc_id in test_doc_ids:
        test_corpus[doc_id] = corpus[doc_id]

    retriever.index_documents(list(test_corpus.values()), list(test_corpus.keys()))
    run = defaultdict(lambda: {})
    for qid, query in tqdm(test_queries.items()):
        results = retriever.search(query, topk=100)
        if rerank:
            document_ids = [r[0] for r in results]
            rerank_scores = reranker.score_documents(queries_dict=test_queries,
                                                     documents_dict=test_corpus,
                                                     queries_ids=qid,
                                                     document_ids=document_ids,
                                                     batch_size=100)
            results = [(doc_id, score[0]) for doc_id, score in zip(document_ids, rerank_scores)]
            results = sorted(results, key=lambda x: x[1], reverse=True)
            results = {str(doc_id): (1/(1+math.exp(-score))) for doc_id, score in results}
        else:
            results = {str(doc_id): score for doc_id, score in results}

        run[str(qid)] = results

    metrics = ["ndcg_cut_10", "ndcg_cut_100",
               "map_cut_10", "map_cut_100",
               "recip_rank",
               "recall_5", "recall_10", "recall_100"]

    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    results = evaluator.evaluate(run)

    for measure in sorted(metrics):
        print(
            measure,
            'all',
            pytrec_eval.compute_aggregated_measure(
                measure,
                [query_measures[measure]
                 for query_measures in results.values()]))


def interactive_demo(retriever, data_path="data", split="test", rerank=False):
    if rerank:
        reranker = ReRanker(model_path="weights_ranker",
                            model_type="distilroberta-base",
                            use_gpu=True,
                            parallel=True,
                            debug=False,
                            max_seq_length=300)
    else:
        reranker = None

    corpus = {}  # Our whole corpus, pid => passage
    corpus_names = {}
    # Read passages
    with open(f"{data_path}/docs.jsonl", encoding="utf8") as f:
        for line in f:
            d = json.loads(line.rstrip("\n"))
            name = d["name"]
            description = d["description"] if d["description"] is not None else ""
            readme = d["readme"] if d["readme"] is not None else ""
            corpus[d["id"]] = " </s> ".join([name, description, readme]).replace("\r\n", " ").replace("\n", " ").replace("  ", " ")
            corpus_names[d["id"]] = d["name"]

    test_corpus = {}
    test_queries = {}  # Our dev queries. qid => query
    qrel = defaultdict(lambda: {})  # Mapping qid => set with relevant pids
    test_doc_ids = set()
    repo_names = set()
    with open(f"{data_path}/qrels_{split}.jsonl", encoding="utf8") as f:
        for line in f:
            d = json.loads(line.rstrip("\n"))
            test_queries[d["qid"]] = d["query"]
            for doc_id in d["docs"]:
                if corpus_names[doc_id] not in repo_names:
                    test_doc_ids.add(doc_id)
                    qrel[str(d["qid"])][str(doc_id)] = 1
                    repo_names.add(corpus_names[doc_id])

    test_doc_ids = list(test_doc_ids)
    for doc_id in test_doc_ids:
        test_corpus[doc_id] = corpus[doc_id]

    retriever.index_documents(list(test_corpus.values()), list(test_corpus.keys()))

    query = input('query: ')
    while query != "exit":
        test_queries["query"] = query
        results = retriever.search(query, topk=20)
        if rerank:
            document_ids = [r[0] for r in results]
            rerank_scores = reranker.score_documents(queries_dict=test_queries,
                                                     documents_dict=test_corpus,
                                                     queries_ids="query",
                                                     document_ids=document_ids,
                                                     batch_size=10)
            results = [(doc_id, score[0]) for doc_id, score in zip(document_ids, rerank_scores)]
            results = sorted(results, key=lambda x: x[1], reverse=True)
            results = [f"{test_corpus[doc_id][:200]} : {(1 / (1 + math.exp(-score)))}" for doc_id, score in results]
        else:
            results = [f"{test_corpus[doc_id][:200]} : {score}" for doc_id, score in results]

        for r in results:
            print(r)

        query = input('query: ')


if __name__ == "__main__":
    retriever = SparseRetriever() #DenseEncoder("weights_pretrained") #Doc2vecRetriever()
    interactive_demo(retriever)
    #evaluate(retriever, split="test")
