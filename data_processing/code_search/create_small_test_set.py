import csv
import os
import random
import hydra

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, Union
from hydra.core.config_store import ConfigStore


@dataclass
class CreateSmallTestSetConfig:
    query_tsv_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/formated_questions_test.tsv'
    qrels_path: Any = (
        '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels.tsv',
        '/work/jkillingback_umass_edu/data/stack-overflow-data/train/qrels.tsv',
        '/work/jkillingback_umass_edu/data/stack-overflow-data/val/qrels.tsv',
    )
    output_dir: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/siddharth_test/'
    basename: str = 'queries_small'
    qrels_basename: str = 'qrels_small'
    split_by_type: bool = True
    seed: int = 1234
    num_queries: int = 2_000


cs = ConfigStore.instance()
cs.store(name='config', node=CreateSmallTestSetConfig)


def read_query_tsv(path: str) -> Dict[Union[str, int], str]:
    output = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for qid, text in reader:
            output[qid] = text
    return output


def filter_queries(queries, text_filter):
    return {k: v for k, v in queries.items() if text_filter in k}


def dict_to_list(dict_input):
    return [(k, v) for k, v in dict_input.items()]


def write_tsv(input: List[Iterable], output_path) -> None:
    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in input:
            writer.writerow(row)


def read_qrels(path):
    output = []
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            output.append(row)
    return output


def filter_qrels(qrels, queries: List[Tuple[str, str]]):
    qid_set = set([qid for qid, query in queries])
    return [row for row in qrels if row[0] in qid_set]


@hydra.main(config_path=None, config_name='config')
def main(cfg: CreateSmallTestSetConfig) -> None:
    queries = read_query_tsv(cfg.query_tsv_path)
    if type(cfg.qrels_path) is not str:
        qrels = []
        for qrel_path in cfg.qrels_path:
            qrels += read_qrels(qrel_path)
    else:
        qrels = read_qrels(cfg.qrels_path)

    random.seed(cfg.seed)
    if cfg.split_by_type:
        title_queries = dict_to_list(filter_queries(queries, 't'))
        question_queries = dict_to_list(filter_queries(queries, 'q'))

        sampled_title_queries = random.sample(title_queries, cfg.num_queries)
        sampled_question_queries = random.sample(question_queries, cfg.num_queries)

        title_qrels = filter_qrels(qrels, sampled_title_queries)
        question_qrels = filter_qrels(qrels, sampled_question_queries)

        title_output_path = os.path.join(cfg.output_dir, cfg.basename + '_titles.tsv')
        question_output_path = os.path.join(
            cfg.output_dir, cfg.basename + '_questions.tsv'
        )

        write_tsv(sampled_title_queries, title_output_path)
        write_tsv(sampled_question_queries, question_output_path)

        title_qrel_output_path = os.path.join(
            cfg.output_dir, cfg.qrels_basename + '_titles.tsv'
        )
        question_qrel_output_path = os.path.join(
            cfg.output_dir, cfg.qrels_basename + '_questions.tsv'
        )

        write_tsv(title_qrels, title_qrel_output_path)
        write_tsv(question_qrels, question_qrel_output_path)
    else:
        queries = dict_to_list(queries)
        sampled_queries = random.sample(queries, cfg.num_queries)
        output_path = os.path.join(cfg.output_dir, cfg.basename + '.tsv')
        write_tsv(sampled_queries, output_path)


if __name__ == '__main__':
    main()
