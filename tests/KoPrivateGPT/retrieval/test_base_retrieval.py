import os
import pathlib
import pickle
from typing import List, Union
from uuid import UUID

from KoPrivateGPT.DB import PickleDB
from KoPrivateGPT.schema import Passage

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

# bwook
TEST_DB_ORIGIN2 = [{
    'db_type': 'mongo_db',
    'db_path': {
        'mongo_url': 'test_url_1',
        'db_name': 'test_db_name_1',
        'collection_name': 'test_collection_name_1'
        }
    },
    {
        'db_type': 'pickle_db',
        'db_path': {
            'save_path': "test"
        }
    },
    {
        'db_type': 'mongo_db',
        'db_path': {
            'mongo_url': 'test_url_2',
            'db_name': 'test_db_name_2',
            'collection_name': 'test_collection_name_2'
        }
    },
        {
        'db_type': 'mongo_db',
        'db_path': {
            'mongo_url': 'test_url_1',
            'db_name': 'test_db_name_1',
            'collection_name': 'test_collection_name_1'
        }
    }
]

TEST_IDS2 = [passage.id for passage in TEST_PASSAGES[:3]]

"""
# For 'duplicate_check'
# Basic : db_origin이 정해진 type = dict[tuple : list]  으로 잘 나오는가?
# 1. 서로 다른 "db_type"을 잘 구분하는가?
# 2. 서로 다른 "db_path"를 잘 구분하는가?
"""

"""
For 'is_created'
1. db_instance_list가 비어있을 때, list에 db instance가 잘 들어가는가? 해당 db를 잘 return하는가?
2. db_instance_list가 비어있지 않고, db_origin이 이미 존재하는 경우, 해당 db를 잘 return하는가?
3. db_instance_list가 비어있지 않고, db_origin이 존재하지 않는 경우, list에 db instance가 잘 들어가는가? 해당 db를 잘 return하는가?
"""

"""
For 'fetch_each_db'
-> 미리 세팅된 db가 필요함, 그리고 그 db_instance를 가져와 data를 fetch.
- dict[tuple : list]와 ids를 받으면 List[Passage]를 return 하는가?
"""