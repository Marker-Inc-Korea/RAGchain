import json
import os
import pathlib
import sys

from RAGchain.retrieval import BM25Retrieval, VectorDBRetrieval
from RAGchain.utils.util import text_modifier
from config import Options
from run_localGPT import select_vectordb

sys.path.append(str(pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent))

import click
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from ingest import REPO_ID


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


@click.command()
@click.option("--test_type", default="dev", help="dev or train")
@click.option("--retrieval_type", default="vectordb", help="retrieval type to use, select vectordb or bm25")
@click.option("--suffix", default="ko-sroberta-multitask", help="suffix for prediction file")
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--embedding_type', default='KoSimCSE',
              help='embedding model to use, select OpenAI or KoSimCSE or ko-sroberta-multitask')
@click.option('--vectordb_type', default='chroma', help='vector database to use, select chroma or pinecone')
def main(test_type, retrieval_type, suffix, device_type, embedding_type, vectordb_type):
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
    # make retrieval
    vectordb = select_vectordb(vectordb_type, embedding_type, device_type)
    if retrieval_type in text_modifier('bm25'):
        retrieval = BM25Retrieval(save_path=Options.bm25_db_dir)
    elif retrieval_type in text_modifier('vectordb'):
        retrieval = VectorDBRetrieval(vectordb=vectordb)
    else:
        raise ValueError("retrieval type is not valid")
    pred = {}
    for key in tqdm(list(data.keys())):
        query = data[key]["question"]
        retrieved_ids = retrieval.retrieve_id(query, top_k=10)
        pred[key] = {
            "answer": str(True),
            "decomposition": [],
            "paragraphs": retrieved_ids
        }

    # save prediction
    with open(f"./{test_type}_pred_{suffix}.json", "w") as f:
        json.dump(pred, f)


if __name__ == "__main__":
    main()
