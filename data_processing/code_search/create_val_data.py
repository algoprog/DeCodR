import csv
import json
import random
import os
import tantivy

from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


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


def index_collection(collection, queries, pids):
    schema_builder = tantivy.SchemaBuilder()
    schema_builder.add_text_field('body', stored=True)
    schema_builder.add_text_field('pid', stored=True)
    schema = schema_builder.build()
    index = tantivy.Index(schema)

    writer = index.writer()
    for document in collection:
        writer.add_document(tantivy.Document(body=[document['body']], pid=[document['pid']]))
    writer.commit()
    index.reload()

    searcher = index.searcher()

    queries = [query for query in queries if 't' in query['qid']]

    for query_vals in queries:
        # print('query:', query_vals['query'][:100])
        try:
            query = index.parse_query(query_vals['query'], ['body'])
        except:
            # print('Could not run query', query_vals['query'][:100])
            continue
        for score, pid in searcher.search(query, 100).hits:
            # print(score, pid)
            doc = searcher.doc(pid)
            pids.add(doc['pid'][0])
        #     print(doc['pid'], doc['body'])
        #     print('-' * 5, end='\n\n')
        # print('\n\n\n')
    
    with open('pids.json', 'w') as f:
        json.dump(list(pids), f)


def qrel_pids(qrels):
    return set([qrel['pid'] for qrel in qrels])

def main():
    collection_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/train/collection.tsv'
    query_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/val/queries.tsv'
    qrel_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/val/qrels.tsv'

    collection = read_csv(collection_path, ['pid', 'body'])
    queries = read_csv(query_path, ['qid', 'query'])
    qrels = read_csv(qrel_path, ['qid', 'version', 'pid', 'rel'])

    pids = qrel_pids(qrels)
    index_collection(collection, queries, pids)


if __name__ == '__main__':
    main()
