import click

from RAGchain.DB import MongoDB, PickleDB
from RAGchain.pipeline import BasicIngestPipeline
from RAGchain.preprocess.loader import FileLoader
from RAGchain.retrieval import BM25Retrieval, VectorDBRetrieval
from RAGchain.utils.util import text_modifier
from config import MongoDBOptions
from config import Options, PickleDBOptions
from run_localGPT import select_vectordb


@click.command()
@click.option('--device_type', default='mps', help='device to run on, select gpu, cpu or mps')
@click.option('--vectordb_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='ko-sroberta-multitask',
              help='embedding model to use, select OpenAI or KoSimCSE or ko-sroberta-multitask')
@click.option('--retrieval_type', default='bm25', help='retrieval type to use, select vectordb or bm25')
@click.option('--db_type', default='mongo_db', help='db type to use, select pickle_db or mongo_db')
def main(device_type, vectordb_type, embedding_type, retrieval_type: str, db_type: str):
    vectordb = select_vectordb(vectordb_type, embedding_type, device_type)
    if retrieval_type in text_modifier('bm25'):
        retrieval = BM25Retrieval(save_path=Options.bm25_db_dir)
    elif retrieval_type in text_modifier('vectordb'):
        retrieval = VectorDBRetrieval(vectordb=vectordb)
    else:
        raise ValueError("retrieval type is not valid")

    if db_type in text_modifier('mongo_db'):
        db = MongoDB(MongoDBOptions.mongo_url,
                     MongoDBOptions.db_name,
                     MongoDBOptions.collection_name)
    elif db_type in text_modifier('pickle_db'):
        db = PickleDB(PickleDBOptions.save_path)
    else:
        raise ValueError("db type is not valid")
    pipeline = BasicIngestPipeline(
        file_loader=FileLoader(Options.source_dir, Options.HwpConvertHost),
        retrieval=retrieval,
        db=db
    )
    pipeline.run()


if __name__ == "__main__":
    main()
