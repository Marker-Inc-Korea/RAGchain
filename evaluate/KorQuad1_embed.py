import json
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
from langchain.vectorstores import Chroma
import os
from chromadb.config import Settings

PERSIST_DIRECTORY = f"{os.path.dirname(os.path.realpath(__file__))}/DB/KorQuad1"
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)

if __name__ == "__main__":
    embeddings = HuggingFaceInstructEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask",
                                               model_kwargs={"device": "cuda"})
    with open("../../KorQuad1.0/Dev-set/KorQuAD_v1.0_dev.json") as f:
        dev_set = json.load(f)

    text_list = []
    for t in tqdm(dev_set['data']):
        for data in t['paragraphs']:
            context = data["context"]
            text_list.append(context)

    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    documents = splitter.create_documents(text_list)
    texts = splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()
