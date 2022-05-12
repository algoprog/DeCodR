import csv
import os
import random
import hydra

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, Union
from hydra.core.config_store import ConfigStore
from clean_body import clean_html

@dataclass
class CreateDuplicateTestSetConfig:
    input_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate_query_data.csv'
    qrels_path: Any = (
        '/work/jkillingback_umass_edu/data/stack-overflow-data/test/qrels.tsv',
        '/work/jkillingback_umass_edu/data/stack-overflow-data/train/qrels.tsv',
        '/work/jkillingback_umass_edu/data/stack-overflow-data/val/qrels.tsv',
    )
    duplicate_queries_output_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/queries'
    duplicate_qrels_output_path: str = '/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/qrels'


cs = ConfigStore.instance()
cs.store(name='config', node=CreateDuplicateTestSetConfig)


def read_csv(
    path: str, row_names: List[str], delimiter: str = '\t'
) -> List[Dict[str, Any]]:
    data = []
    with open(path) as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i, row in enumerate(reader):
            # if i > 1_00_000:
            #     break
            entry = {}
            for name, value in zip(row_names, row):
                entry[name] = value
            data.append(entry)
    return data


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
    output = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            output[row[0][:-2]] = row
    return output


def filter_qrels(qrels, queries: List[Tuple[str, str]]):
    qid_set = set([qid for qid, query in queries])
    return [row for row in qrels if row[0] in qid_set]


@hydra.main(config_path=None, config_name='config')
def main(cfg: CreateDuplicateTestSetConfig) -> None:
    row_names = 'OId,OCreationDate,OTitle,OBody,OTags,DCreationDate,DId,DTitle,DBody,DTags'.split(',')
    duplicate_data = read_csv(cfg.input_path, row_names, delimiter=',')
    if type(cfg.qrels_path) is not str:
        qrels = {}
        for qrel_path in cfg.qrels_path:
            qrels.update(read_qrels(qrel_path))
    else:
        qrels = read_qrels(cfg.qrels_path)


    for postfix in ['_t', '_q']:
        duplicate_qrels = []
        duplicate_queries = []
        for data in duplicate_data:
            if data['DId'] in qrels:
                
                    dup_qrel = qrels[data['DId']].copy()
                    new_qid = data['OId'] + postfix
                    dup_qrel[0] = new_qid
                    duplicate_qrels.append(dup_qrel)
                    query_text = data['OTitle'] if 't' in postfix else clean_html(data['OBody'])
                    duplicate_queries.append([new_qid, query_text])

        write_tsv(duplicate_qrels, cfg.duplicate_qrels_output_path + postfix + '.tsv')
        write_tsv(duplicate_queries, cfg.duplicate_queries_output_path + postfix + '.tsv')

if __name__ == '__main__':
    main()
