import csv
import json
import random
import os

from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any


def read_csv(path: str, row_names: List[str]) -> List[Dict[str, Any]]:
    data = []
    with open(path) as f:
        reader = csv.reader(f)
        for row in reader:
            entry = {}
            for name, value in zip(row_names, row):
                entry[name] = value
            data.append(entry)
    return data


def answers_to_collection(answers, collection_path, pids):
    filtered_answers = sorted(
        [
            [answer['id'], answer['body']]
            for answer in answers if answer['id'] in pids
        ]
    )
    print(len(filtered_answers))
    print(type(answers[0]['id']), type(list(pids)[0]))
    with open(collection_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for i, answer in enumerate(filtered_answers):
            if i % 10_000 == 0:
                print(i)
            writer.writerow(answer)


def answers_to_qrels(answers, qrels_path, qids):
    filtered_answers = sorted(
        [
            [answer['id'], answer['questionId']]
            for answer in answers if answer['questionId'] in qids
        ]
    )
    qrels = []
    for answer in filtered_answers:
        qrels.append([answer[1] + '_t', 0, answer[0], 1])
        qrels.append([answer[1] + '_q', 0, answer[0], 1])

    with open(qrels_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for qrel in qrels:
            writer.writerow(qrel)


def questions_to_queries(raw_questions, query_path, qids):
    questions = []

    for question in raw_questions:
        if question['id'] in qids:
            questions.append([question['id'] + '_t', question['title']])
            questions.append([question['id'] + '_q', question['body']])

    with open(query_path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for question in questions:
            writer.writerow(question)


def qid_split(raw_questions, val_size: int = 2_000, test_size: int = 10_000, seed = 1234):

    qids = set([question['id'] for question in raw_questions])
    qids = sorted(list(qids))

    random.seed(seed)
    random.shuffle(qids)

    return {
        'val': qids[:val_size],
        'test': qids[val_size:test_size + val_size],
        'train': qids[test_size + val_size:],
    }


def main():
    question_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/large_questions_clean.csv'
    answer_path = '/work/jkillingback_umass_edu/data/stack-overflow-data/large_answers_clean.csv'
    output_dir = '/work/jkillingback_umass_edu/data/stack-overflow-data/'

    # question_rows = [
    #     'id',
    #     'title',
    #     'tags',
    #     'body',
    #     'acceptedAnswerId',
    #     'score',
    #     'views',
    # ]
    # raw_questions = read_csv(question_path, question_rows)
    # split = qid_split(raw_questions)

    # for k1 in ['val', 'test', 'train']:
    #     for k2 in ['val', 'test', 'train']:
    #         if k1 != k2:
    #             print(k1, k2)
    #             print(len(set(split[k1]).intersection(set(split[k2]))))

    answer_rows = ['id', 'questionId', 'body', 'score']
    answers =  read_csv(answer_path, answer_rows)



    # for k in ['val', 'test', 'train']:
    #     print(f'On {k}')
    #     qids = set(split[k])
    #     shared_dir = os.path.join(output_dir, k)
    #     Path(shared_dir).mkdir(parents=True, exist_ok=True)
    #     questions_to_queries(raw_questions, os.path.join(shared_dir, 'queries.tsv'), qids)
    #     answers_to_qrels(answers, os.path.join(shared_dir, 'qrels.tsv'), qids)
    # shared_dir = os.path.join(output_dir, 'train')
    # answers_to_collection(answers, os.path.join(shared_dir, 'collection.tsv'))

    with open('pids.json') as f:
        pids = json.load(f)

    shared_dir = os.path.join(output_dir, 'val')
    answers_to_collection(answers, os.path.join(shared_dir, 'collection.tsv'), set(pids))

if __name__ == '__main__':
    main()