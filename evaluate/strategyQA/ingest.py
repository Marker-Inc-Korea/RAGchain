from huggingface_hub import hf_hub_download
import sys
import os
import pathlib

sys.path.append(str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent))
from KoPrivateGPT.embed import Embedding
from KoPrivateGPT.retrieval import BM25Retrieval, VectorDBRetrieval
import pandas as pd
from langchain.schema import Document
import click

REPO_ID = "NomaDamas/Ko-StrategyQA"
SAVE_PATH = "./ko-strategy-qa_paragraphs_bm25.pkl"
"""
    This is ingest code only for Ko-StrategyQA
    You can think this code is just one of implementation of this library. 
    If you want to try other retrievers, you can edit main function.
"""


def get_paragraph():
    paragraph_path = hf_hub_download(repo_id=REPO_ID, filename="ko-strategy-qa_paragraphs.parquet", repo_type="dataset")
    paragraph_df = pd.read_parquet(paragraph_path)
    return paragraph_df


def make_document(row):
    return Document(page_content=row["ko-content"], metadata={"id": row["key"]})


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE',
              help='embedding model to use, select OpenAI or KoSimCSE or ko-sroberta-multitask')
@click.option('--retriever_type', default='vectordb', help='retriever type to use, select vectordb or bm25')
def main(device_type, db_type, embedding_type, retriever_type):
    paragraph = get_paragraph()
    paragraph["document"] = paragraph.apply(make_document, axis=1)
    documents = paragraph["document"].tolist()
    if retriever_type in ['bm25', 'BM25']:
        retriever = BM25Retrieval.load(SAVE_PATH)
        retriever.save(documents)
        retriever.persist(SAVE_PATH)
    else:
        embeddings = Embedding(embed_type=embedding_type, device_type=device_type)
        retriever = VectorDBRetrieval.load(db_type=db_type, embedding=embeddings)
        retriever.save(documents)
    print("DONE")


if __name__ == "__main__":
    main()
