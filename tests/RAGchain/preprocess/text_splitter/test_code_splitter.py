import copy

import pytest
from langchain.schema import Document

from RAGchain.preprocess.text_splitter import CodeSplitter

# from test_documents.test_document_for_code_splitter import TEST_DOCUMENT_CODE_SPLITEER

# Here is test documents with all language that splitter can split for test.
# PYTHON_TEST_DOCUMENT = TEST_DOCUMENT_CODE_SPLITEER.PYTHON_TEST_DOCUMENT()

'''
JS_TEST_DOCUMENT = TEST_DOCUMENT_CODE_SPLITEER.JS_TEST_DOCUMENT()

TS_TEST_DOCUMENT = TEST_DOCUMENT_CODE_SPLITEER.TS_TEST_DOCUMENT()

Markdown_TEST_DOCUMENT = TEST_DOCUMENT_CODE_SPLITEER.Markdown_TEST_DOCUMENT()

Latex_TEST_DOCUMENT = TEST_DOCUMENT_CODE_SPLITEER.Latex_TEST_DOCUMENT()

HTML_TEST_DOCUMENT = TEST_DOCUMENT_CODE_SPLITEER.HTML_TEST_DOCUMENT()

Solidity_TEST_DOCUMNET = TEST_DOCUMENT_CODE_SPLITEER.Solidity_TEST_DOCUMNET()

Csharp_TEST_DOCUMENT = TEST_DOCUMENT_CODE_SPLITEER.Csharp_TEST_DOCUMENT()

'''
# Which file do you want to test ?
TEST_DOCUMENT = Document(
    page_content="""
                            def hello_world():
                                print("Hello, World!")
                            
                            # Call the function
                            hello_world()
                                """,
    metadata={
        'source': 'test_source',
        # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
        'Data information': 'test for python code document',
        'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#python'
    }
)


@pytest.fixture
def codesplitter():
    codesplitter = CodeSplitter()
    yield codesplitter


def test_code_splitter(codesplitter):
    passages = codesplitter.split_document(TEST_DOCUMENT)

    assert len(passages) > 1
    assert passages[0].next_passage_id == passages[1].id
    assert passages[1].previous_passage_id == passages[0].id
    assert passages[0].filepath == 'test_source'
    assert passages[0].filepath == passages[1].filepath
    assert passages[0].previous_passage_id is None
    assert passages[-1].next_passage_id is None

    # Check first passage whether it contains header information of fist layout(first div).
    assert ('학박사님을 아세유? 학교가는 동규형 근데 리뷰할때 동규형이 보면 어떡하지') in passages[0].content

    # Check splitter preserve other metadata in original document.
    test_document_metadata = list(copy.deepcopy(TEST_DOCUMENT).metadata.items())
    test_document_metadata.pop(0)
    for element in test_document_metadata:
        assert element in list(passages[1].metadata_etc.items())

    # Check passages' metadata_etc
    ## metadata_etc can't contain file path(Except first part of first div).
    assert ('source', 'test_source') not in list(passages[1].metadata_etc.items())
    assert ('source', 'test_source') not in list(passages[-1].metadata_etc.items())

    # Check HTML header information put in metadata_etc right form.
    assert ('Header 1', '학박사님을 아세유?') in list(passages[1].metadata_etc.items())

    assert ('Header 1', '맨까송') in list(passages[-1].metadata_etc.items())
    assert ('Header 2', '감빡이') in list(passages[-1].metadata_etc.items())
    assert ('Header 3', '근데 ragchain 쓰는 사람이 맨유팬이면 어떡하지') in list(passages[-1].metadata_etc.items())
