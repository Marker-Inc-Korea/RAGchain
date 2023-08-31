import os
import pathlib
import pickle
from typing import List, Union
from uuid import UUID

from KoPrivateGPT.schema import Passage

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
with open(os.path.join(root_dir, "resources", "sample_passages.pkl"), 'rb') as r:
    TEST_PASSAGES = pickle.load(r)


def test_load_passage():
    assert len(TEST_PASSAGES) > 0
    for passage in TEST_PASSAGES:
        assert isinstance(passage, Passage)
        assert isinstance(passage.id, UUID) or isinstance(passage.id, str)


def validate_ids(retrieved_ids: List[Union[str, UUID]], top_k: int):
    assert len(retrieved_ids) == top_k
    for _id in retrieved_ids:
        assert isinstance(_id, str) or isinstance(_id, UUID)
        fetch_result = list(filter(lambda x: getattr(x, 'id') == _id, TEST_PASSAGES))
        assert len(fetch_result) == 1
        assert fetch_result[0].id == _id


def validate_passages(retrieved_passage: List[Passage], top_k: int):
    assert len(retrieved_passage) == top_k
    original_ids = [passage.id for passage in TEST_PASSAGES]
    original_contents = [passage.content for passage in TEST_PASSAGES]
    for passage in retrieved_passage:
        assert isinstance(passage, Passage)
        assert passage.id in original_ids
        assert passage.content in original_contents
        assert passage.next_passage_id in original_ids or passage.next_passage_id is None
        assert passage.previous_passage_id in original_ids or passage.previous_passage_id is None


TEST_DB_ORIGIN = [{
    'db_type': 'mongo_db',
    'db_path': {
        'mongo_url': 'mongodb://localhost:27017',
        'db_name': 'test',
        'collection_name': 'test'
    }
}]

TEST_IDS = [passage.id for passage in TEST_PASSAGES]
