import os
import pathlib

import pytest

from RAGchain.preprocess.loader import DeepdoctectionPDFLoader

data = [{'page_number': 1,
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

answer = [{'Table': 'table_test_1', 'Page_number': 1},
          {'Table': 'table_test_2', 'Page_number': 1},
          {'Title': 'Visconde: Multi-document QA with GPT-3 and Neural Reranking', 'Text': 'TEST_0', 'PageNumber': 1},
          {'Title': '1 Introduction', 'Text': 'TEST_1', 'PageNumber': 1},
          {'Title': '1 Introduction', 'Text': 'TEST_2', 'PageNumber': 2},
          {'Title': '2 Related Work', 'Text': 'TEST_3', 'PageNumber': 2},
          {'Title': '3 Our Method: Visconde', 'Text': 'TEST_4', 'PageNumber': 2},
          {'Title': '3 Our Method: Visconde', 'Text': 'TEST_5', 'PageNumber': 3},
          {'Table': 'table_test_3', 'Page_number': 4},
          {'Title': '3 Our Method: Visconde', 'Text': 'TEST_6', 'PageNumber': 4},
          {'Title': '4 Experiments', 'Text': '', 'PageNumber': 4},
          {'Title': '4.1 IIRC', 'Text': 'TEST_7', 'PageNumber': 4}]

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
