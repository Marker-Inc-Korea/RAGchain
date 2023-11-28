import os
import pathlib

import pytest

from RAGchain.preprocess.loader import DeepdoctectionPDFLoader

data = [
    {'page_number': 0,
     'title': [],
     'text': 'TEST_EMPTY_TITLE',
     'table': []},

    {'page_number': 1,
     'title': ['1 Introduction', 'Visconde: Multi-document QA with GPT-3 and Neural Reranking'],
     'text': 'Visconde: Multi-document QA with GPT-3 and Neural Reranking\nTEST_0\n1 Introduction\nTEST_1',
     'table': ['table_test_1', 'table_test_2']},

    {'page_number': 2,
     'title': ['2 Related Work', '3 Our Method: Visconde'],
     'text': 'TEST_2\n2 Related Work\nTEST_3\n3 Our Method: Visconde\nTEST_4',
     'table': []},

    {'page_number': 3,
     'title': [],
     'text': 'TEST_5',
     'table': []},

    {'page_number': 4,
     'title': ['4.1 IIRC', '4 Experiments'],
     'text': 'TEST_6\n4 Experiments\n4.1 IIRC\nTEST_7',
     'table': ['table_test_3']}]

answer = [{'title': '', 'text': 'TEST_EMPTY_TITLE', 'page_number': 0},
          {'table': 'table_test_1', 'page_number': 1},
          {'table': 'table_test_2', 'page_number': 1},
          {'title': 'Visconde: Multi-document QA with GPT-3 and Neural Reranking', 'text': 'TEST_0', 'page_number': 1},
          {'title': '1 Introduction', 'text': 'TEST_1', 'page_number': 1},
          {'title': '1 Introduction', 'text': 'TEST_2', 'page_number': 2},
          {'title': '2 Related Work', 'text': 'TEST_3', 'page_number': 2},
          {'title': '3 Our Method: Visconde', 'text': 'TEST_4', 'page_number': 2},
          {'title': '3 Our Method: Visconde', 'text': 'TEST_5', 'page_number': 3},
          {'table': 'table_test_3', 'page_number': 4},
          {'title': '3 Our Method: Visconde', 'text': 'TEST_6', 'page_number': 4},
          {'title': '4 Experiments', 'text': '', 'page_number': 4},
          {'title': '4.1 IIRC', 'text': 'TEST_7', 'page_number': 4}]

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
file_dir = os.path.join(root_dir, "resources", "ingest_files")


@pytest.fixture
def deepdoctection_pdf_loader():
    pdf_filepath = os.path.join(file_dir, 'test1.pdf')
    assert os.path.exists(pdf_filepath)
    assert bool(os.getenv('DEEPDOCTECTION_HOST'))
    deepdoctection_pdf_loader = DeepdoctectionPDFLoader(file_path=pdf_filepath,
                                                        deepdoctection_host=os.getenv('DEEPDOCTECTION_HOST'))
    yield deepdoctection_pdf_loader


def test_extract_pages(deepdoctection_pdf_loader):
    result = data
    extracted_pages = deepdoctection_pdf_loader.extract_pages(result)
    assert extracted_pages == answer


def test_deepdoctection_pdf_loader(deepdoctection_pdf_loader):
    docs = deepdoctection_pdf_loader.load()
    print(len(docs))
    assert len(docs) == 15
    assert docs[-1].metadata['page_number'] == 3
    table_count_content = sum('<table>' in doc.page_content for doc in docs)
    table_count_metadata = sum('table' in doc.metadata['page_type'] for doc in docs)
    assert table_count_content == table_count_metadata == 3
    text_count_content = sum('text' in doc.page_content for doc in docs)
    text_count_metadata = sum('text' in doc.metadata['page_type'] for doc in docs)
    assert text_count_content == text_count_metadata == 12
