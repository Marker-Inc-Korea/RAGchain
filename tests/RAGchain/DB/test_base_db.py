from datetime import datetime
from typing import List

from RAGchain.DB.base import BaseDB
from RAGchain.schema import Passage

TEST_PASSAGES: List[Passage] = [
    Passage(
        id='test_id_1',
        content='This is test number 1',
        filepath='./test/first_file.txt',
        content_datetime=datetime(2022, 2, 3),
        previous_passage_id=None,
        next_passage_id='test_id_2',
        metadata_etc={'test': 'test1'}
    ),
    Passage(
        id='test_id_2',
        content='This is test number 2',
        filepath='./test/second_file.txt',
        content_datetime=datetime(2022, 2, 4),
        previous_passage_id='test_id_1',
        next_passage_id='test_id_3',
        metadata_etc={'test': 'test2'}
    ),
    Passage(
        id='test_id_3',
        content='This is test number 3',
        filepath='./test/second_file.txt',
        content_datetime=datetime(2022, 2, 5),
        previous_passage_id='test_id_2',
        next_passage_id='test_id_4',
        metadata_etc={'test': 'test3'}
    ),
    Passage(
        id='test_id_4',
        content='This is test number 3',
        filepath='./test/third_file.txt',
        content_datetime=datetime(2022, 3, 6),
        previous_passage_id='test_id_3',
        next_passage_id=None,
        metadata_etc={'test': 'test3'}
    )
]


def fetch_test_base(instance: BaseDB):
    fetch_passages = instance.fetch([passage.id for passage in TEST_PASSAGES])
    assert len(fetch_passages) == len(TEST_PASSAGES)
    for passage in fetch_passages:
        assert passage in TEST_PASSAGES

    one_passage = instance.fetch([TEST_PASSAGES[0].id])
    assert one_passage[0].is_exactly_same(TEST_PASSAGES[0])


def search_test_base(db: BaseDB):
    test_result_1 = db.search(filepath=['./test/second_file.txt'])
    assert len(test_result_1) == 2
    assert 'test_id_2' in [passage.id for passage in test_result_1]
    assert 'test_id_3' in [passage.id for passage in test_result_1]

    test_result_2 = db.search(test=['test1'])
    assert len(test_result_2) == 1
    assert 'test_id_1' == test_result_2[0].id

    test_result_3 = db.search(test=['test1', 'test2'])
    assert len(test_result_3) == 2
    assert 'test_id_1' in [passage.id for passage in test_result_3]
    assert 'test_id_2' in [passage.id for passage in test_result_3]

    test_result_4 = db.search(content=['This is test number 2'], filepath=['./test/second_file.txt'])
    assert len(test_result_4) == 1
    assert 'test_id_2' == test_result_4[0].id

    test_result_5 = db.search(id=['test_id_3', 'test_id_4'])
    assert len(test_result_5) == 2
    assert 'test_id_3' in [passage.id for passage in test_result_5]
    assert 'test_id_4' in [passage.id for passage in test_result_5]

    test_result_6 = db.search(id=['test_id_3', 'test_id_4'], filepath=['./test/second_file.txt'])
    assert len(test_result_6) == 1
    assert 'test_id_3' == test_result_6[0].id

    test_result_7 = db.search(content_datetime_range=[(datetime(2022, 3, 1), datetime.now())])
    assert len(test_result_7) == 1
    assert 'test_id_4' == test_result_7[0].id

    test_result_8 = db.search(content_datetime_range=[(datetime(2022, 2, 1), datetime(2022, 2, 10))],
                              content=['This is test number 3'])
    assert len(test_result_8) == 1
    assert 'test_id_3' == test_result_8[0].id
