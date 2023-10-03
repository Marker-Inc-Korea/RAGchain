import os
import pathlib

import pytest

from KoPrivateGPT.preprocess.loader import NougatPDFLoader

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
file_dir = os.path.join(root_dir, "resources", "ingest_files")


@pytest.fixture
def nougat_pdf_loader():
    pdf_filepath = os.path.join(file_dir, 'test1.pdf')
    assert os.path.exists(pdf_filepath)
    assert bool(os.getenv('NOUGAT_HOST'))
    nougat_pdf_loader = NougatPDFLoader(file_path=pdf_filepath, nougat_host=os.getenv('NOUGAT_HOST'))
    yield nougat_pdf_loader


def test_nougat_pdf_loader(nougat_pdf_loader):
    docs = nougat_pdf_loader.load()
    assert len(docs) == 17
    for doc in docs:
        assert bool(doc.page_content)
        if doc.metadata["content_type"] == "table":
            assert doc.page_content.startswith("\\\\begin{table}")
        else:
            assert bool(doc.page_content)

    docs = nougat_pdf_loader.load(split_section=True, split_table=False,
                                  start=1, stop=2)
    assert len(docs) == 9
    for doc in docs:
        assert bool(doc.page_content)
        assert doc.metadata["content_type"] == "text"
