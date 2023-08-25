import os

import pytest

import test_base_retrieval

chroma_path = os.path.join(test_base_retrieval.root_dir, "resources", "chroma")


@pytest.fixture
def vectordb_retrieval():
    # TODO : make vectorDB retrieval test after close Feature/#110
    pass
