import os
from typing import List, Tuple

import click
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from KoPrivateGPT.embed import Embedding
from KoPrivateGPT.options import Options
from KoPrivateGPT.retrieval import VectorDBRetrieval
from KoPrivateGPT.retrieval import BM25Retrieval
from KoPrivateGPT.loader import FileLoader


def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE',
              help='embedding model to use, select OpenAI or KoSimCSE or ko-sroberta-multitask')
@click.option('--retriever_type', default='vectordb', help='retriever type to use, select vectordb or bm25')
def main(device_type, db_type, embedding_type, retriever_type):
    load_dotenv()

    # Load documents and split in chunks
    print(f"Loading documents from {Options.source_dir}")
    file_loader = FileLoader(Options.source_dir)
    documents = file_loader.load()

    if len(documents) <= 0:
        print(f"Could not find any new documents in {Options.source_dir}")
        return
    texts = split_documents(documents)
    print(f"Loaded {len(documents)} documents from {Options.source_dir}")
    print(f"Split into {len(texts)} chunks of text")

    if retriever_type in ['bm25', 'BM25']:
        retriever = BM25Retrieval.load(Options.bm25_db_dir)
    else:
        embeddings = Embedding(embed_type=embedding_type, device_type=device_type)
        retriever = VectorDBRetrieval.load(db_type=db_type, embedding=embeddings)

    retriever.save(texts)
    if retriever_type in ['bm25', 'BM25']:
        retriever.persist(Options.bm25_db_dir)


if __name__ == "__main__":
    main()
