from typing import List

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage

TEST_PASSAGES: List[Passage] = [
    Passage(
        id='test_id_1',
        content='This is test number 1',
        filepath='./test/first_file.txt',
        previous_passage_id=None,
        next_passage_id='test_id_2',
        metadata_etc={'test': 'test1'}
    ),
    Passage(
        id='test_id_2',
        content='This is test number 2',
        filepath='./test/second_file.txt',
        previous_passage_id='test_id_1',
        next_passage_id='test_id_3',
        metadata_etc={'test': 'test2'}
    ),
    Passage(
        id='test_id_3',
        content='This is test number 3',
        filepath='./test/second_file.txt',
        previous_passage_id='test_id_2',
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
