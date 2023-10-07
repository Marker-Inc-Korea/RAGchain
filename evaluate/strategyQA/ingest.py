import os
import pathlib
import sys

sys.path.append(str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent))
import click
from KoPrivateGPT.preprocess.loader import KoStrategyQALoader
from KoPrivateGPT.retrieval import VectorDBRetrieval, BM25Retrieval
from KoPrivateGPT.utils.util import text_modifier
from config import Options
from run_localGPT import select_vectordb
from KoPrivateGPT.pipeline.basic import BasicDatasetPipeline

REPO_ID = "NomaDamas/Ko-StrategyQA"
SAVE_PATH = "./ko-strategy-qa_paragraphs_bm25.pkl"
"""
    This is ingest code only for Ko-StrategyQA
    You can think this code is just one of implementation of this library. 
"""


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--vectordb_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='ko-sroberta-multitask',
              help='embedding model to use, select OpenAI or KoSimCSE or ko-sroberta-multitask')
@click.option('--retrieval_type', default='vectordb', help='retriever type to use, select vectordb or bm25')
def main(device_type, vectordb_type, embedding_type, retrieval_type):
    vectordb = select_vectordb(vectordb_type, embedding_type, device_type)
    if retrieval_type in text_modifier('bm25'):
        retrieval = BM25Retrieval(save_path=Options.bm25_db_dir)
    elif retrieval_type in text_modifier('vectordb'):
        retrieval = VectorDBRetrieval(vectordb=vectordb)
    else:
        raise ValueError("retrieval type is not valid")
    pipeline = BasicDatasetPipeline(file_loader=KoStrategyQALoader(),
                                    retrieval=retrieval)
    pipeline.run()
    print("DONE")


if __name__ == "__main__":
    main()
