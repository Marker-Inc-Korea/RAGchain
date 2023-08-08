import json
from typing import List
import sys
import os
import pathlib

sys.path.append(str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent))

import click
from huggingface_hub import hf_hub_download
from langchain.schema import Document
from tqdm import tqdm

from KoPrivateGPT.embed import Embedding
from ingest import SAVE_PATH, REPO_ID
from KoPrivateGPT.retrieve import BM25Retriever, VectorDBRetriever


def get_train():
    train_json = hf_hub_download(repo_id=REPO_ID, filename="ko-strategy-qa_train.json", repo_type="dataset")
    with open(train_json, "r") as f:
        train = json.load(f)
    return train


def get_dev():
    dev_json = hf_hub_download(repo_id=REPO_ID, filename="ko-strategy-qa_dev.json", repo_type="dataset")
    with open(dev_json, "r") as f:
        dev = json.load(f)
    return dev


def extract_keys(documents: List[Document]):
    return [doc.metadata["id"] for doc in documents]


@click.command()
@click.option("--test_type", default="dev", help="dev or train")
@click.option("--retriever_type", default="vectordb", help="retriever type to use, select vectordb or bm25")
@click.option("--suffix", default="ko-sroberta-multitask", help="suffix for prediction file")
def main(test_type, retriever_type, suffix):
    """
        This script allows you to test data retrieval using the BM25Retriever model.
        By default, the test type is 'dev'. You can specify the test type by using the '--test_type' option.
    """
    # get data
    if test_type == "dev":
        data = get_dev()
    elif test_type == "train":
        data = get_train()
    else:
        raise ValueError("test_type should be dev or train")
    # make retrieve
    if retriever_type in ['bm25', 'BM25']:
        retriever = BM25Retriever.load(SAVE_PATH)
    else:
        embeddings = Embedding(embed_type='ko-sroberta-multitask', device_type='mps')
        # llm = load_model("openai")
        # embeddings = hyde_embeddings(llm, embeddings)
        retriever = VectorDBRetriever.load(db_type='chroma', embedding=embeddings)
    pred = {}
    for key in tqdm(list(data.keys())):
        query = data[key]["question"]
        retrieved_documents = retriever.retrieve(query, top_k=10)
        pred[key] = {
            "answer": str(True),
            "decomposition": [],
            "paragraphs": extract_keys(retrieved_documents)
        }

    # save prediction
    with open(f"./{test_type}_pred_{suffix}.json", "w") as f:
        json.dump(pred, f)


if __name__ == "__main__":
    main()
