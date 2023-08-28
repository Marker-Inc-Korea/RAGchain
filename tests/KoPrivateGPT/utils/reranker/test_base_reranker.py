import os
import pathlib
import pickle
from uuid import UUID

from KoPrivateGPT.schema import Passage

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
with open(os.path.join(root_dir, "resources", "sample_passages.pkl"), 'rb') as r:
    TEST_PASSAGES = pickle.load(r)


def test_load_passage():
    assert len(TEST_PASSAGES) > 0
    for passage in TEST_PASSAGES:
        assert isinstance(passage, Passage)
        assert isinstance(passage.id, UUID) or isinstance(passage.id, str)
