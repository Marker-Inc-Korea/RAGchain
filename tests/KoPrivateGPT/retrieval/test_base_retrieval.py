import os
import pathlib
import pickle
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


def validate_ids(retrieved_ids, top_k):
    assert len(retrieved_ids) == top_k
    for _id in retrieved_ids:
        assert isinstance(_id, str) or isinstance(_id, UUID)
        fetch_result = list(filter(lambda x: getattr(x, 'id') == _id, TEST_PASSAGES))
        assert len(fetch_result) == 1
        assert fetch_result[0].id == _id


TEST_DB_ORIGIN = [{
    'db_type': 'mongo_db',
    'db_path': {
        'mongo_url': 'mongodb://localhost:27017',
        'db_name': 'test',
        'collection_name': 'test'
    }
}]

TEST_IDS = [passage.id for passage in TEST_PASSAGES]