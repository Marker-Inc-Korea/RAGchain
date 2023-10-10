import os
import pathlib
import pickle
from uuid import UUID

from RAGchain.DB import PickleDB
from RAGchain.schema import Passage
from RAGchain.retrieval import BM25Retrieval

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
with open(os.path.join(root_dir, "resources", "sample_passages.pkl"), 'rb') as r:
    TEST_PASSAGES = pickle.load(r)


def test_load_passage():
    assert len(TEST_PASSAGES) > 0
    for passage in TEST_PASSAGES:
        assert isinstance(passage, Passage)
        assert isinstance(passage.id, UUID) or isinstance(passage.id, str)


def ready_pickle_db(pickle_path: str):
    db = PickleDB(save_path=pickle_path)
    db.create_or_load()
    db.save(TEST_PASSAGES)
    return db


def ready_bm25_retrieval(bm25_path: str):
    retrieval = BM25Retrieval(save_path=bm25_path)
    retrieval.ingest(TEST_PASSAGES)
    return retrieval


def validate_answer(answer: str, passages: list, passage_cnt: int = 4):
    assert bool(answer)
    assert len(passages) == passage_cnt

    solution_ids = [passage.id for passage in TEST_PASSAGES]
    for passage in passages:
        assert passage.id in solution_ids
