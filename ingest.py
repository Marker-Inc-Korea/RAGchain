import os

import click
from typing import List

from embed.embedded_files_cache import EmbeddedFilesCache
from utils import xlxs_to_csv
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from hwp import HwpLoader
from vectorDB import DB
from dotenv import load_dotenv
from tqdm import tqdm
from options import Options
from embed.embedding import Embedding

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


def load_documents(source_dir: str) -> List[Document]:
    # Loads all documents from source documents directory
    docs = []
    file_cache = EmbeddedFilesCache()
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
    file_cache.save()
    return docs


def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    return texts


def ingest_texts(device_type, db_type, embedding_type, texts):
    # Create embeddings
    embeddings = Embedding(embed_type=embedding_type, device_type=device_type).embedding()
    db = DB(db_type, embeddings).from_documents(texts)
    db = None


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE', help='embedding model to use, select OpenAI or KoSimCSE')
def main(device_type, db_type, embedding_type):
    load_dotenv()

    #  Load documents and split in chunks
    print(f"Loading documents from {Options.source_dir}")
    documents = load_documents(Options.source_dir)

    if len(documents) <= 0:
        print(f"Could not find any new documents in {Options.source_dir}")
        return
    texts = split_documents(documents)
    print(f"Loaded {len(documents)} documents from {Options.source_dir}")
    print(f"Split into {len(texts)} chunks of text")

    ingest_texts(device_type, db_type, embedding_type, texts)


if __name__ == "__main__":
    main()
