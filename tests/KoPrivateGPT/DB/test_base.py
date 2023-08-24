from typing import List

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.schema import Passage

test_passages: List[Passage] = [
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


def test_fetch(instance: BaseDB):
    instance.save(test_passages)
    fetch_passages = instance.fetch([passage.id for passage in test_passages])
    assert len(fetch_passages) == len(test_passages)
    for passage in fetch_passages:
        assert passage in test_passages

    one_passage = instance.fetch([test_passages[0].id])
    assert one_passage[0].is_exactly_same(test_passages[0])
