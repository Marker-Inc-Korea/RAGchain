import pytest

from RAGchain.preprocess.loader.pdf_link_loader import PdfLinkLoader


@pytest.fixture
def pdf_link_loader():
    yield PdfLinkLoader("https://www.africau.edu/images/default/sample.pdf")


def test_pdf_link_loader(pdf_link_loader):
    docs = pdf_link_loader.load()
    assert len(docs) == 1
    assert docs[0].page_content.strip().startswith("A Simple PDF File")


def test_valid_url():
    assert PdfLinkLoader.valid_url("http://www.google.com") is True
    assert PdfLinkLoader.valid_url("https://www.google.com") is True
    assert PdfLinkLoader.valid_url("www.google.com") is False
    assert PdfLinkLoader.valid_url("google.com") is False
    assert PdfLinkLoader.valid_url("google") is False
