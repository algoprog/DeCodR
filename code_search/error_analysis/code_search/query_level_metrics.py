import argparse
from collections import defaultdict
from email.policy import default
from enum import Enum
import json
from pathlib import Path
import hydra
import os
import sys

import pytrec_eval

from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import Union, Dict, Any, Tuple, List, Optional


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
class QueryMetricEvalConfig:
    run_path: str = '/work/jkillingback_umass_edu/DeCodR/code_search/baseline/bm25_small_questions_run.txt'
    # run_path: str = '/work/jkillingback_umass_edu/code_search_runs/20_000_small_test_questions.txt'
    model: ModelTypes = ModelTypes.BM25
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels_small_titles.tsv'
    output_dir: str = '/work/jkillingback_umass_edu/DeCodR/error_analysis/code_search/query_metrics'
    dataset: DataSetTypes = DataSetTypes.DUPLICATE_QUESTIONS
    

@dataclass
class DuplicateQuestionsConfig(QueryMetricEvalConfig):
    dataset = DataSetTypes.DUPLICATE_QUESTIONS
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/qrels_q.tsv'


@dataclass
class DuplicateTitlesConfig(QueryMetricEvalConfig):
    dataset = DataSetTypes.DUPLICATE_TITLES
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/qrels_t.tsv'


@dataclass
class SmallQuestionsConfig(QueryMetricEvalConfig):
    dataset = DataSetTypes.SMALL_QUESTIONS
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels_small_questions.tsv'


@dataclass
class SmallTitlesConfig(QueryMetricEvalConfig):
    dataset = DataSetTypes.SMALL_TITLES
    qrels_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels_small_titles.tsv'



cs = ConfigStore.instance()
cs.store(name='config', node=SmallQuestionsConfig)


@hydra.main(config_path=None, config_name='config')
def main(cfg: QueryMetricEvalConfig) -> None:
    assert os.path.exists(cfg.qrels_path)
    assert os.path.exists(cfg.run_path)

    with open(cfg.qrels_path, 'r') as f_qrel:
        qrel = pytrec_eval.parse_qrel(f_qrel)

    with open(cfg.run_path, 'r') as f_run:
        run = pytrec_eval.parse_run(f_run)

    evaluator = pytrec_eval.RelevanceEvaluator(
        qrel, pytrec_eval.supported_measures)

    results = evaluator.evaluate(run)

    def print_line(measure, scope, value):
        print('{:25s}{:8s}{:.4f}'.format(measure, scope, value))

    measures = ['recip_rank', 'recall_100']
    output = defaultdict(dict)
    for query_id, query_measures in sorted(results.items()):
        for measure in measures:
            output[measure][query_id] = query_measures[measure]


    output_dir = os.path.join(cfg.output_dir, cfg.dataset.name.lower())
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = os.path.join(output_dir, cfg.model.name.lower() + '.json')
    print(output_path)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)    


if __name__ == '__main__':
    main()
