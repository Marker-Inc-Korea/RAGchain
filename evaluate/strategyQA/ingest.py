import os
import pathlib
import sys

from KoPrivateGPT.preprocess.loader import KoStrategyQALoader

sys.path.append(str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent))
import click

from KoPrivateGPT.pipeline.basic import BasicDatasetPipeline
from KoPrivateGPT.utils.embed import EmbeddingFactory


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
    pipeline = BasicDatasetPipeline(file_loader=KoStrategyQALoader(),
                                    retrieval_type=(retrieval_type, {"save_path": SAVE_PATH,
                                                                     "vectordb_type": vectordb_type,
                                                                     "embedding": EmbeddingFactory(
                                                                         embed_type=embedding_type,
                                                                         device_type=device_type).get()
                                                                     }))
    pipeline.run()
    print("DONE")


if __name__ == "__main__":
    main()
