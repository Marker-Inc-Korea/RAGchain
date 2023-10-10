import glob
import os
import pathlib

import pytest

from RAGchain.preprocess.loader import FileLoader, KoStrategyQALoader

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
file_dir = os.path.join(root_dir, "resources", "ingest_files")


@pytest.fixture
def file_loader():
    assert os.path.exists(file_dir)
    assert len(glob.glob(file_dir)) > 0
    file_loader = FileLoader(target_dir=file_dir, hwp_host_url=os.getenv('HWP_CONVERTER_HOST'))
    yield file_loader


@pytest.fixture
def ko_strategy_qa_loader():
    ko_strategy_qa_loader = KoStrategyQALoader()
    yield ko_strategy_qa_loader


def test_file_loader(file_loader):
    for ext in file_loader.ingestable_extensions:
        docs = file_loader.load(filter_ext=[ext])
        assert len(docs) > 0
        assert bool(docs[0].page_content) is True


def test_ko_strategy_qa_loader(ko_strategy_qa_loader):
    docs = ko_strategy_qa_loader.load()
    assert len(docs) > 0
    assert bool(docs[0].page_content) is True
