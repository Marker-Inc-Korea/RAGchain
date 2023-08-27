"""
Warnings! This file is deprecated.
We will add proper way to delete retrieval embeddings, DB contents, linker files, etc.
TODO : Add proper way to delete retrieval embeddings, DB contents, linker files, etc. - Feature/#118
"""

import os
import shutil

import click
import pinecone

from KoPrivateGPT.options import ChromaOptions, PineconeOptions, Options
from KoPrivateGPT.utils.util import FileChecker


@click.command()
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--retriever_type', default='vectordb', help='retriever type to use, select vectordb or bm25')
def main(db_type, retriever_type):
    raise RuntimeError("This function is deprecated and not working properly. We will solve this issue at Feature/#118")
    if retriever_type in ['bm25', 'BM25']:
        delete_sparse_retrieval_index(Options.bm25_db_dir)
    else:
        delete_embeddings_vectordb(db_type)


def delete_embeddings_vectordb(db_type):
    raise RuntimeError("This function is deprecated and not working properly. We will solve this issue at Feature/#118")
    if db_type in ['chroma', 'Chroma', 'CHROMA']:
        shutil.rmtree(str(ChromaOptions.persist_dir))
    elif db_type in ['pinecone', 'Pinecone', 'PineCone', 'PINECONE']:
        pinecone.delete_index(PineconeOptions.index_name)
    else:
        raise ValueError(f"Unknown db type: {db_type}")
    print(f"Deleted {db_type} embeddings")


def delete_sparse_retrieval_index(save_path: str):
    raise RuntimeError("This function is deprecated and not working properly. We will solve this issue at Feature/#118")
    if not FileChecker(save_path).check_type(file_types=['.pkl', '.pickle']).is_exist():
        print(f'Could not find sparse retrieval index: {save_path}')
        return
    os.remove(save_path)
    print(f"Deleted sparse retrieval index: {save_path}")


if __name__ == "__main__":
    main()
