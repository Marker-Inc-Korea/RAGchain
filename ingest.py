from typing import List

import click
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from KoPrivateGPT.DB.pickle_db import PickleDB
from KoPrivateGPT.text_splitter import RecursiveTextSplitter
from KoPrivateGPT.utils.embed import Embedding
from KoPrivateGPT.options import Options, DBOptions
from KoPrivateGPT.retrieval import VectorDBRetrieval
from KoPrivateGPT.retrieval import BM25Retrieval
from KoPrivateGPT.loader import FileLoader


def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--vectordb_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE',
              help='embedding model to use, select OpenAI or KoSimCSE or ko-sroberta-multitask')
@click.option('--retrieval_type', default='vectordb', help='retrieval type to use, select vectordb or bm25')
def main(device_type, vectordb_type, embedding_type, retrieval_type):
    load_dotenv()

    # File Loader
    print(f"Loading documents from {Options.source_dir}")
    file_loader = FileLoader(Options.source_dir)
    documents = file_loader.load()
    if len(documents) <= 0:
        print(f"Could not find any new documents in {Options.source_dir}")
        return
    print(f"Loaded {len(documents)} documents from {Options.source_dir}")

    # Text Splitter
    splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
    passages = []
    for document in documents:
        passages.extend(splitter.split_document(document))
    print(f"Split into {len(passages)} passages")

    # Save passages to DB
    db = PickleDB(DBOptions.save_path)
    db.create_or_load()
    db.save(passages)

    # Ingest to retrieval
    if retrieval_type in ['bm25', 'BM25']:
        retrieval = BM25Retrieval(Options.bm25_db_dir, db=db)
    else:
        embeddings = Embedding(embed_type=embedding_type, device_type=device_type)
        retrieval = VectorDBRetrieval(vectordb_type=vectordb_type, embedding=embeddings, db=db)
    retrieval.ingest(passages)


if __name__ == "__main__":
    main()
