from collections import defaultdict
import csv
from enum import Enum
import hashlib
import json
import os
from pathlib import Path
import hydra
import matplotlib.pyplot as plt
import numpy as np

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Union, Dict, Any, Tuple, List, Optional
from spacy.lang.en import English
from rouge_score import rouge_scorer


class DataSetTypes(Enum):
    DUPLICATE_QUESTIONS = 1
    DUPLICATE_TITLES = 2
    SMALL_QUESTIONS = 3
    SMALL_TITLES = 4
    VALIDATION = 5


class ModelTypes(Enum):
    BM25 = 1
    CO_0 = 2
    CO_10_000 = 3
    CO_20_000 = 4
    CO_30_000 = 5


@dataclass
class AnalysisConfig:
    models: Tuple[ModelTypes] = (ModelTypes.CO_20_000, ModelTypes.BM25)
    metric_dir: str = '/work/jkillingback_umass_edu/DeCodR/error_analysis/code_search/query_metrics'
    collection_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/train/collection.tsv'
    queries_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/queries_q.tsv'
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/qrels_q.tsv'
    dataset: DataSetTypes = DataSetTypes.DUPLICATE_QUESTIONS
    cache: str = '/work/jkillingback_umass_edu/cache'


@dataclass
class DuplicateQuestionsConfig(AnalysisConfig):
    dataset = DataSetTypes.DUPLICATE_QUESTIONS
    queries_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/queries_q.tsv'
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/qrels_q.tsv'


@dataclass
class DuplicateTitlesConfig(AnalysisConfig):
    dataset = DataSetTypes.DUPLICATE_TITLES
    queries_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/queries_t.tsv'
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/qrels_t.tsv'


@dataclass
class SmallQuestionsConfig(AnalysisConfig):
    dataset = DataSetTypes.SMALL_QUESTIONS
    queries_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/queries_small_questions.tsv'
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels_small_questions.tsv'


@dataclass
class SmallTitlesConfig(AnalysisConfig):
    dataset = DataSetTypes.SMALL_TITLES
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels_small_titles.tsv'
    queries_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/queries_small_titles.tsv'


cs = ConfigStore.instance()
cs.store(name='config', node=SmallQuestionsConfig)


def read_csv(
    path: str, row_names: List[str], delimiter: str = '\t', limit=None
) -> List[Dict[str, Any]]:
    data = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            if limit is not None and i > limit:
                break
            entry = {}
            for name, value in zip(row_names, row):
                entry[name] = value
            data.append(entry)
    return data


def largest_distance(combined_data, metric_key):
    model_names = list(combined_data.keys())
    distances = []

    data1 = combined_data[model_names[0]]
    data2 = combined_data[model_names[1]]
    for qid, value1 in data1[metric_key].items():
        value2 = data2[metric_key][qid]
        distances.append(
            (value1 - value2, value1, value2, model_names[0], model_names[1], qid)
        )

    return sorted(distances, reverse=False)


def largest_distance_calculations(largest_dists, collection, queries, qrels):
    nlp = English()
    tokenizer = nlp.tokenizer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    sums = {
        'largest_seq': defaultdict(int),
        'recall': defaultdict(int),
        'answer_length': defaultdict(int),
        'question_length': defaultdict(int),
    }
    stores = {
        'largest_seq': defaultdict(list),
        'recall': defaultdict(list),
        'answer_length': defaultdict(list),
        'question_length': defaultdict(list),
    }
    counts = defaultdict(int)
    for dist, score1, score2, model1, model2, qid in largest_dists:
        query = queries[qid]
        answer = collection[qrels[qid]]
        tokenized_answer = [tok.text for tok in tokenizer(answer)]
        tokenized_query = [tok.text for tok in tokenizer(query)]
        scores = scorer.score(query, answer)
        largest_sequence = rouge_scorer._lcs_table(tokenized_answer, tokenized_query)[
            -1
        ][-1]

        sums['largest_seq'][round(dist, 1)] += largest_sequence
        sums['recall'][round(dist, 1)] += scores['rougeL'].recall
        sums['answer_length'][round(dist, 1)] += len(tokenized_answer)
        sums['question_length'][round(dist, 1)] += len(tokenized_query)

        stores['largest_seq'][round(dist, 1)].append(largest_sequence)
        stores['recall'][round(dist, 1)].append(scores['rougeL'].recall)
        stores['answer_length'][round(dist, 1)].append(len(tokenized_answer))
        stores['question_length'][round(dist, 1)].append(len(tokenized_query))
        counts[round(dist, 1)] += 1

    for k in sums.keys():
        for dist in sums[k].keys():
            sums[k][dist] /= counts[dist]

    data = defaultdict(list)
    labels = []
    for k in sums.keys():
        print(k)
        for dist, avg in sorted([(dist, avg) for dist, avg in sums[k].items()]):
            print(dist, round(avg, 3), np.std(stores[k][dist]), np.mean(stores[k][dist]))
            data[k].append(stores[k][dist])
            if k == 'recall':
                labels.append(dist)
        print('\n\n\n')

    metric = 'Recall@100'
    kwargs = {
        'largest_seq': {'ylim': (0, 250), 'xlabel': f'Difference in {metric}', 'ylabel': 'Largest Shared Sequence Length'},
        'recall': {'ylim': (0, 0.8), 'xlabel': f'Difference in {metric}', 'ylabel': 'RougueL Recall'},
        'answer_length': {'ylim': (0, 1200), 'xlabel': f'Difference in {metric}', 'ylabel': '# of Answer Tokens'},
        'question_length': {'ylim': (0, 1200), 'xlabel': f'Difference in {metric}', 'ylabel': '# of Question Tokens'},
    }

    plt.rc('axes', labelsize=12)
    for k in sums.keys():
        fig, ax = plt.subplots(figsize=(15,10))
        vp = ax.violinplot(data[k], [i for i in range(0, len(data[k]) * 3, 3)], widths=2,
                        showmeans=True, showmedians=False, showextrema=False)
        # styling:
        for body in vp['bodies']:
            body.set_alpha(0.9)
        vp['cmeans'].set_color('#000000')
        ax.set(xticks=np.arange(0, len(data[k]) * 3, 3), xticklabels=labels, **kwargs[k])

        plt.savefig(f'/work/jkillingback_umass_edu/DeCodR/error_analysis/code_search/figs/{k}.png')
        plt.show()


def print_largest_distance_output(largest_dists, collection, queries, qrels):
    nlp = English()
    tokenizer = nlp.tokenizer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)

    for dist, score1, score2, model1, model2, qid in largest_dists[:10]:
        query = queries[qid]
        answer = collection[qrels[qid]]
        tokenized_answer = [tok.text for tok in tokenizer(answer)]
        tokenized_query = [tok.text for tok in tokenizer(query)]

        # overlap = len(tokenized_answer.intersection(tokenized_query))
        # overlap /= len(tokenized_answer.union(tokenized_query))
        scores = scorer.score(query, answer)
        largest_sequence = rouge_scorer._lcs_table(tokenized_answer, tokenized_query)[
            -1
        ][-1]
        print(
            f'{dist} {largest_sequence} {scores}  {model1}={score1}, {model2}={score2}'
        )
        print(query)
        print('*' * 50)
        print(answer)
        print('-' * 50, end='\n\n\n')

    print('=' * 50, end='\n\n\n')
    print('=' * 50)
    for dist, score1, score2, model1, model2, qid in largest_dists[-10:]:
        query = queries[qid]
        answer = collection[qrels[qid]]
        tokenized_answer = [tok.text for tok in tokenizer(answer)]
        tokenized_query = [tok.text for tok in tokenizer(query)]

        # overlap = len(tokenized_answer.intersection(tokenized_query))
        # overlap /= len(tokenized_answer.union(tokenized_query))
        scores = scorer.score(query, answer)
        largest_sequence = rouge_scorer._lcs_table(tokenized_answer, tokenized_query)[
            -1
        ][-1]
        print(
            f'{dist} {largest_sequence} {scores}  {model1}={score1}, {model2}={score2}'
        )
        print(query)
        print('*' * 50)
        print(answer)
        print('-' * 50, end='\n\n\n')


@hydra.main(config_path=None, config_name='config')
def main(cfg: AnalysisConfig) -> None:
    nlp = English()
    tokenizer = nlp.tokenizer

    combined_data = {}
    for model in cfg.models:
        input_dir = os.path.join(cfg.metric_dir, cfg.dataset.name.lower())
        input_path = os.path.join(input_dir, model.name.lower() + '.json')

        with open(input_path) as f:
            combined_data[model.name.lower()] = json.load(f)

    queries = read_csv(cfg.queries_path, ['qid', 'text'])
    query_lookup = {query['qid']: query['text'] for query in queries}
    print('Done loading queries')

    qrels = read_csv(cfg.qrels_path, ['qid', '0', 'pid', 'relevance'])
    qrels_lookup = {qrel['qid']: qrel['pid'] for qrel in qrels}
    pid_set = set([qrel['pid'] for qrel in qrels])
    print('Done loading qrels')

    collection_cache = os.path.join(cfg.cache, str(hashlib.sha1(cfg.qrels_path.encode("utf-8")).hexdigest()) + '.json')
    print(collection_cache)
    if os.path.isfile(collection_cache):
        print('Loading cache...')
        with open(collection_cache) as f:
            collection_lookup = json.load(f)
    else:
        collection = read_csv(cfg.collection_path, ['pid', 'text'])
        collection_lookup = {passage['pid']: passage['text'] for passage in collection if passage['pid'] in pid_set}
        with open(collection_cache, 'w') as f:
            json.dump(collection_lookup, f)

    largest_dists = largest_distance(combined_data, 'recall_100')
    largest_distance_calculations(
        largest_dists, collection_lookup, query_lookup, qrels_lookup
    )
    # print_largest_distance_output(largest_dists, collection_lookup, query_lookup, qrels_lookup)


if __name__ == '__main__':
    main()
