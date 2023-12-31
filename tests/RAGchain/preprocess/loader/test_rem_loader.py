import os
import pathlib
from datetime import datetime

import pytest

from RAGchain.preprocess.loader.rem_loader import RemLoader

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
rem_path = os.path.join(root_dir, "resources", "rem_sample.sqlite3")


@pytest.fixture
def rem_loader():
    loader = RemLoader(rem_path)
    time_range_loader = RemLoader(rem_path, time_range=[datetime(2023, 12, 31, 15, 9, 0), datetime.now()])
    yield loader, time_range_loader


def test_rem_loader(rem_loader):
    result = rem_loader[0].load()
    assert len(result) == 39

    result = rem_loader[1].load()
    assert len(result) == 5

    assert bool(result[0].page_content) is True
    assert isinstance(result[0].metadata['content_datetime'], datetime)
