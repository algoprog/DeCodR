import logging
import os
import tantivy

from shutil import rmtree


class SparseRetriever:
    def __init__(self, path='sparse_index', load=True, reset=True):
        if reset:
            try:
                rmtree(path)
                logging.info("Cleared previous index")
            except:
                pass
        if not os.path.exists(path):
            os.mkdir(path)
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("body", stored=False)
        schema_builder.add_unsigned_field("doc_id", stored=True)
        schema = schema_builder.build()
        self.index = tantivy.Index(schema, path=path, reuse=load)
        self.searcher = self.index.searcher()

    def index_documents(self, documents, ids=None):
        logging.info('Building sparse index of {} docs...'.format(len(documents)))
        writer = self.index.writer()
        for i, doc in enumerate(documents):
            doc_id = i
            if ids is not None:
                doc_id = ids[i]
            writer.add_document(tantivy.Document(
                body=[doc],
                doc_id=doc_id
            ))
            if (i + 1) % 100000 == 0:
                writer.commit()
                logging.info('Indexed {} docs'.format(i + 1))
        writer.commit()
        logging.info('Built sparse index')
        self.index.reload()
        self.searcher = self.index.searcher()

    def search(self, query, topk=100):
        query = self.index.parse_query(query, ["body"])
        scores = self.searcher.search(query, topk).hits
        results = [(self.searcher.doc(doc_id)['doc_id'][0], score)
                   for score, doc_id in scores]
        return results


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)

    retriever = SparseRetriever()

    from evaluate import evaluate
    evaluate(retriever, split="test")
