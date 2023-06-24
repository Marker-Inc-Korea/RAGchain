import os
import click
from typing import List
from utils import xlxs_to_csv
from langchain.document_loaders import TextLoader, PDFMinerLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS, SOURCE_DIRECTORY, PERSIST_DIRECTORY
from langchain.embeddings import HuggingFaceInstructEmbeddings
from hwp import HwpLoader
from db import DB
from tqdm import tqdm

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
    for (path, dir, files) in tqdm(os.walk(source_dir)):
        for file_path in files:
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == '.xlsx':
                for doc in xlxs_to_csv(os.path.join(path, file_path)):
                    docs.append(load_single_document(doc))
            elif ext in ['.txt', '.pdf', '.csv', '.hwp']:
                docs.append(load_single_document(os.path.join(path, file_path)))
            else:
                print(f"Unknown file type: {file_path}")
    return docs


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
def main(device_type, db_type):
    # load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']:
        device='cpu'
    elif device_type in ['mps', 'MPS']:
        device='mps'
    else:
        device='cuda'

    #  Load documents and split in chunks
    print(f"Loading documents from {SOURCE_DIRECTORY}")
    documents = load_documents(SOURCE_DIRECTORY)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    print(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    print(f"Split into {len(texts)} chunks of text")

    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask",
                                               model_kwargs={"device": device})

    db = DB(db_type, embeddings).from_documents(texts)
    db = None


if __name__ == "__main__":
    main()
