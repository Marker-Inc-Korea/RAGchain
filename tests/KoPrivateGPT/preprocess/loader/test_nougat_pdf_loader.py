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
    assert os.getenv('NOUGAT_HOST')
    nougat_pdf_loader = NougatPDFLoader(file_path=pdf_filepath, nougat_host=os.getenv('NOUGAT_HOST'))
    yield nougat_pdf_loader


def test_nougat_pdf_loader(nougat_pdf_loader):
    doc = nougat_pdf_loader.load()
