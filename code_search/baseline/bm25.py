import csv
from pyserini.search.lucene import LuceneSearcher

from typing import Dict, Union


def do_search(searcher, queries, output_path):
    queries =  [(id, text) for (id, text) in queries.items()]
    qids, query_text = list(zip(*queries))

    query_text = list(query_text)
    qids = list(qids)

    print('Started search')
    hits = searcher.batch_search(queries=query_text, qids=qids, k=1_000)
    print('Search complete')

    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for qid, results in hits.items():
            for i in range(len(results)):
                # print([qid, results[i].docid, i, results[i].score, 0])
                writer.writerow([qid, 'Q0', results[i].docid, i + 1, results[i].score, 'bm25'])
    exit()

    with open(output_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for query_num, (qid, query) in enumerate(queries):
            if query_num % 10_000 == 0:
                print(query_num)
            # print(query)
            hits = searcher.search(query, k=1_000)
            for i in range(len(hits)):
                writer.writerow([qid, hits[i].docid, i, hits[i].score, 0])
                # print(f'{i+1:2} {hits[i].docid:4} {hits[i].score:.5f}')


def read_tsv(path: str) -> Dict[Union[str, int], str]:
    output = {}
    with open(path) as f:
        reader = csv.reader(f, delimiter='\t')
        for id, text in reader:
            output[id] = text
    return output


def fix_run(existing_path, new_path):
    output = []
    with open(existing_path) as f:
        reader = csv.reader(f, delimiter='\t')
        for qid, docid, rank, score, tag in reader:
            output.append([qid, 'Q0', docid, int(rank) + 1, score, 'bm25'])

    with open(new_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for row in output:
            writer.writerow(row)


def main():
    searcher = LuceneSearcher('/work/jkillingback_umass_edu/indexes/stack-overflow-index')
    queries = read_tsv('/work/jkillingback_umass_edu/data/stack-overflow-data/duplicate-test/queries_q.tsv')
    do_search(searcher, queries, 'bm25_duplicate_questions_run.txt')
    # fix_run('bm25_small_questions_run.txt', 'bm25_small_questions_run_fixed.txt')
    


if __name__ == '__main__':
    main()