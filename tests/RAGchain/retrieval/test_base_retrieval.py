import os
import pathlib
import pickle
from datetime import datetime
from typing import List, Union
from uuid import UUID

import pytest

from RAGchain.DB import PickleDB, MongoDB
from RAGchain.retrieval import BM25Retrieval
from RAGchain.schema import Passage

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
with open(os.path.join(root_dir, "resources", "sample_passages.pkl"), 'rb') as r:
    TEST_PASSAGES = pickle.load(r)

SEARCH_TEST_PASSAGES: List[Passage] = [
    Passage(
        id='test_id_1_search',
        content='This is test number 1',
        filepath='./test/first_file.txt',
        content_datetime=datetime(2021, 1, 1),
        previous_passage_id=None,
        next_passage_id='test_id_2',
        metadata_etc={'test': 'test1'}
    ),
    Passage(
        id='test_id_2_search',
        content='This is test number 2',
        filepath='./test/second_file.txt',
        content_datetime=datetime(2021, 1, 2),
        previous_passage_id='test_id_1',
        next_passage_id='test_id_3',
        metadata_etc={'test': 'test2'}
    ),
    Passage(
        id='test_id_3_search',
        content='This is test number 3',
        filepath='./test/second_file.txt',
        content_datetime=datetime(2021, 1, 3),
        previous_passage_id='test_id_2',
        next_passage_id='test_id_4',
        metadata_etc={'test': 'test3'}
    ),
    Passage(
        id='test_id_4_search',
        content='This is test number 3',
        filepath='./test/third_file.txt',
        content_datetime=datetime(2021, 11, 4),
        previous_passage_id='test_id_3',
        next_passage_id=None,
        metadata_etc={'test': 'test3'}
    )
]


def test_load_passage():
    assert len(TEST_PASSAGES) > 0
    for passage in TEST_PASSAGES:
        assert isinstance(passage, Passage)
        assert isinstance(passage.id, UUID) or isinstance(passage.id, str)


def ready_pickle_db(pickle_path: str):
    db = PickleDB(save_path=pickle_path)
    db.create_or_load()
    db.save(TEST_PASSAGES)
    db.save(SEARCH_TEST_PASSAGES)
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


# Below is the Feature #156
TEST_DB_ORIGIN = [{
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
            'save_path': "test.pkl"
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

TEST_DB_ORIGIN_RESULT = {(('db_type', 'mongo_db'), ('db_path', (('mongo_url', 'test_url_1'),
                                                                ('db_name', 'test_db_name_1'),
                                                                ('collection_name', 'test_collection_name_1')))): [0,
                                                                                                                   3],
                         (('db_type', 'pickle_db'), ('db_path', (('save_path', 'test.pkl'),))): [1],
                         (('db_type', 'mongo_db'), ('db_path', (('mongo_url', 'test_url_2'),
                                                                ('db_name', 'test_db_name_2'),
                                                                ('collection_name', 'test_collection_name_2')))): [2]}


@pytest.fixture
def just_bm25_retrieval():
    bm25_path = os.path.join(root_dir, "resources", "bm25", "just_bm25_retrieval.pkl")
    pickle_path = os.path.join(root_dir, "resources", "pickle", "just_bm25_retrieval.pkl")
    if not os.path.exists(os.path.dirname(bm25_path)):
        os.makedirs(os.path.dirname(bm25_path))
    bm25_retrieval = BM25Retrieval(save_path=bm25_path)
    ready_pickle_db(pickle_path)
    yield bm25_retrieval
    # teardown
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
    if os.path.exists(pickle_path):
        os.remove(pickle_path)


def test_duplicate_check(just_bm25_retrieval):
    assert just_bm25_retrieval.duplicate_check(TEST_DB_ORIGIN) == TEST_DB_ORIGIN_RESULT


def test_is_created(just_bm25_retrieval):
    """
    For 'is_created'
    1. db_instance_list가 비어있을 때, list에 db instance가 잘 들어가는가? 해당 db를 잘 return하는가?
    2. db_instance_list가 비어있지 않고, db_origin이 이미 존재하는 경우, 해당 db를 잘 return하는가?
    3. db_instance_list가 비어있지 않고, db_origin이 존재하지 않는 경우, list에 db instance가 잘 들어가는가? 해당 db를 잘 return하는가?
    """
    # 0. reset db_instance_list for test
    just_bm25_retrieval.db_instance_list = []
    # 1. If 'db_instance_list' is empty
    first_instance = just_bm25_retrieval.is_created(db_type=TEST_DB_ORIGIN[0]['db_type'],
                                                    db_path=TEST_DB_ORIGIN[0]['db_path'])
    assert first_instance == just_bm25_retrieval.db_instance_list[0]
    # 2. db_instance_list is not empty, db_origin already exists
    second_instance = just_bm25_retrieval.is_created(db_type=TEST_DB_ORIGIN[0]['db_type'],
                                                     db_path=TEST_DB_ORIGIN[0]['db_path'])
    assert second_instance == just_bm25_retrieval.db_instance_list[0]
    assert second_instance == first_instance
    # 3. db_instance_list is not empty, db_origin does not already exist
    third_instance = just_bm25_retrieval.is_created(db_type=TEST_DB_ORIGIN[1]['db_type'],
                                                    db_path=TEST_DB_ORIGIN[1]['db_path'])
    assert third_instance == just_bm25_retrieval.db_instance_list[1]


TEST_DB_ORIGIN_RESULT_2 = {(('db_type', 'mongo_db'),
                            ('db_path', (('mongo_url', f'{os.getenv("MONGO_URL")}'),
                                         ('db_name', f'{os.getenv("MONGO_DB_NAME")}'),
                                         ('collection_name', 'test_retrieval'))))
                           : [0],
                           (('db_type', 'mongo_db'),
                            ('db_path', (('mongo_url', f'{os.getenv("MONGO_URL")}'),
                                         ('db_name', f'{os.getenv("MONGO_DB_NAME")}'),
                                         ('collection_name', 'test_retrieval_2'))))
                           : [1]
                           }

TEST_PASSAGES_2 = [TEST_PASSAGES[0]]
TEST_PASSAGES_3 = [TEST_PASSAGES[1]]
TEST_RESULT_PASSAGES = [TEST_PASSAGES[0], TEST_PASSAGES[1]]

TEST_IDS = [TEST_PASSAGES[0].id, TEST_PASSAGES[1].id]


@pytest.fixture
def just_mongo_db():
    mongo_db = MongoDB(
        mongo_url=os.getenv('MONGO_URL'),
        db_name=os.getenv('MONGO_DB_NAME'),
        collection_name='test_retrieval')
    mongo_db.create_or_load()
    yield mongo_db
    # teardown
    mongo_db.collection.drop()
    assert mongo_db.collection_name not in mongo_db.db.list_collection_names()


@pytest.fixture
def just_mongo_db_2():
    mongo_db = MongoDB(
        mongo_url=os.getenv('MONGO_URL'),
        db_name=os.getenv('MONGO_DB_NAME'),
        collection_name='test_retrieval_2')
    mongo_db.create_or_load()
    yield mongo_db
    # teardown
    mongo_db.collection.drop()
    assert mongo_db.collection_name not in mongo_db.db.list_collection_names()


def test_fetch_each_db(just_mongo_db, just_mongo_db_2, just_bm25_retrieval):
    # Create db_instance
    just_mongo_db.save(TEST_PASSAGES_2)
    # Create another db_instance
    just_mongo_db_2.save(TEST_PASSAGES_3)
    # Test
    assert just_bm25_retrieval.fetch_each_db(TEST_DB_ORIGIN_RESULT_2, TEST_IDS) == TEST_RESULT_PASSAGES
