import os
import pickle
from datetime import datetime

import click
from typing import List
from utils import xlxs_to_csv
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from hwp import HwpLoader
from db import DB
from dotenv import load_dotenv
from tqdm import tqdm
from embedding import EMBEDDING
from options import Options

HwpConvertOpt = 'all'#'main-only'
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
    embedded_files = get_embedded_files_cache()
    for (path, dir, files) in tqdm(os.walk(source_dir)):
        for file_name in files:
            ext = os.path.splitext(file_name)[-1].lower()
            full_file_path = os.path.join(path, file_name)
            try:
                _ = embedded_files[full_file_path]
                continue
            except:
                pass
            if ext == '.xlsx':
                embedded_files[full_file_path] = datetime.now()
                for doc in xlxs_to_csv(full_file_path):
                    docs.append(load_single_document(doc))
            elif ext in ['.txt', '.pdf', '.csv', '.hwp']:
                embedded_files[full_file_path] = datetime.now()
                docs.append(load_single_document(full_file_path))
            else:
                print(f"Not Support file type {ext} yet.")
    save_embedded_files_cache(embedded_files)
    return docs


def get_embedded_files_cache():
    # Load the embedded files cache
    if os.path.exists(Options.embedded_files_cache_dir):
        with open(Options.embedded_files_cache_dir, 'rb') as f:
            embedded_files_cache = pickle.load(f)
    else:
        embedded_files_cache = {}
    return embedded_files_cache


def save_embedded_files_cache(embedded_files_cache):
    # Save the embedded files cache
    with open(Options.embedded_files_cache_dir, 'wb') as f:
        pickle.dump(embedded_files_cache, f)


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE', help='embedding model to use, select OpenAI or KoSimCSE')
def main(device_type, db_type, embedding_type):
    load_dotenv()
    # load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']:
        device = 'cpu'
    elif device_type in ['mps', 'MPS']:
        device = 'mps'
    else:
        device = 'cuda'

    #  Load documents and split in chunks
    print(f"Loading documents from {Options.source_dir}")
    documents = load_documents(Options.source_dir)
    if len(documents) <= 0:
        print(f"Could not find any new documents in {Options.source_dir}")
        return
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {Options.source_dir}")
    print(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = EMBEDDING(embed_type=embedding_type).embedding()

    db = DB(db_type, embeddings).from_documents(texts)
    db = None


if __name__ == "__main__":
    main()
