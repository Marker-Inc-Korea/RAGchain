import os
from typing import List, Tuple

import click
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from embed.embedded_files_cache import EmbeddedFilesCache
from embed.embedding import Embedding
from hwp import HwpLoader
from options import Options
from retrieve import LangchainRetriever
from retrieve import BM25Retriever
from utils import xlxs_to_csv

HwpConvertOpt = 'all'  # 'main-only'
HwpConvertHost = f'http://hwp-converter:7000/upload?option={HwpConvertOpt}'


def load_single_document(file_path: str) -> Document:
    # Loads a single document from a file path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf8")
    elif file_path.endswith(".pdf"):
        loader = PDFMinerLoader(file_path)
    elif file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.endswith(".hwp"):
        loader = HwpLoader(file_path, hwp_convert_path=HwpConvertHost)

    return loader.load()[0]


def load_documents(source_dir: str, file_cache: EmbeddedFilesCache = None) -> Tuple[List[Document], EmbeddedFilesCache]:
    # Loads all documents from source documents directory
    if file_cache is None:
        file_cache = EmbeddedFilesCache()
    docs = []
    for (path, dir, files) in tqdm(os.walk(source_dir)):
        for file_name in files:
            ext = os.path.splitext(file_name)[-1].lower()
            full_file_path = os.path.join(path, file_name)
            if file_cache.is_exist(full_file_path):
                continue
            if ext == '.xlsx':
                file_cache.add(full_file_path)
                for doc in xlxs_to_csv(full_file_path):
                    docs.append(load_single_document(doc))
            elif ext in ['.txt', '.pdf', '.csv', '.hwp']:
                file_cache.add(full_file_path)
                docs.append(load_single_document(full_file_path))
            else:
                print(f"Not Support file type {ext} yet.")
    return docs, file_cache


def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE', help='embedding model to use, select OpenAI or KoSimCSE')
@click.option('--retriever_type', default='langchain', help='retriever type to use, select langchain or bm25')
def main(device_type, db_type, embedding_type, retriever_type):
    load_dotenv()

    #  Load documents and split in chunks
    print(f"Loading documents from {Options.source_dir}")
    documents, file_cache = load_documents(Options.source_dir)

    if len(documents) <= 0:
        print(f"Could not find any new documents in {Options.source_dir}")
        return
    texts = split_documents(documents)
    print(f"Loaded {len(documents)} documents from {Options.source_dir}")
    print(f"Split into {len(texts)} chunks of text")

    if retriever_type in ['bm25', 'BM25']:
        retriever = BM25Retriever.load(Options.bm25_db_dir)
    else:
        embeddings = Embedding(embed_type=embedding_type, device_type=device_type).embedding()
        retriever = LangchainRetriever.load(db_type=db_type, embedding=embeddings)

    retriever.save(texts)

    file_cache.save()


if __name__ == "__main__":
    main()
