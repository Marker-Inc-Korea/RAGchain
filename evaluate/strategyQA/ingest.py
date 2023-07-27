from huggingface_hub import hf_hub_download
from retrieve import BM25Retriever
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


def main():
    paragraph = get_paragraph()
    paragraph["document"] = paragraph.apply(make_document, axis=1)
    documents = paragraph["document"].tolist()
    retriever = BM25Retriever.load(SAVE_PATH)
    retriever.save(documents)
    retriever.persist(SAVE_PATH)


if __name__ == "__main__":
    main()
