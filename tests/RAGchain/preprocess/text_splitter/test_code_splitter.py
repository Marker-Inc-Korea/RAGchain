import copy
import pytest
from langchain.schema import Document

from RAGchain.preprocess.text_splitter import CodeSplitter

# Which file do you want to test ?
# PYTHON test
python_doc = Document(
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


# JS test

JS_doc = Document(
page_content="""
function helloWorld()
{
    console.log("Hello, World!");
}

// Call the function
helloWorld();
""",
metadata={
    'source': 'test_source',
    # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
    'Data information': 'test for js code document',
    'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#js',
}
)


# C# test
Csharp_doc = Document(
    page_content="""
using System;
class Program
{
    static void Main()
    {
        int age = 30; // Change the age value as needed

        // Categorize the age without any console output
        if (age < 18)
        {
            // Age is under 18
        }
        else if (age >= 18 && age < 65)
        {
            // Age is an adult
        }
        else
        {
            // Age is a senior citizen
        }
    }
}
"""
    ,
    metadata={
        'source': 'test_source',
        # Check whether the metadata_etc contains the multiple information from the TEST DOCUMENT metadatas or not.
        'Data information': 'test for C# text document',
        'Data reference link': 'https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/code_splitter#c',
    }
)




# Test doucment define.
python_TEST_DOCUMENT = python_doc
JS_doc_TEST_DOCUMENT = JS_doc
csharp_TEST_DOCUMENT = Csharp_doc
test_documents = [python_TEST_DOCUMENT, JS_doc_TEST_DOCUMENT, csharp_TEST_DOCUMENT]


@pytest.fixture
def python_code_splitter():
    python_code_splitter = CodeSplitter(language_name= 'PYTHON', chunk_size= 50, chunk_overlap= 0)
    yield python_code_splitter

@pytest.fixture
def JS_code_splitter():
    JS_code_splitter = CodeSplitter(language_name= 'JS', chunk_size= 60, chunk_overlap= 0)
    yield JS_code_splitter

@pytest.fixture
def csharp_code_splitter():
    csharp_code_splitter = CodeSplitter(language_name= 'CSHARP', chunk_size= 17, chunk_overlap= 0)
    yield csharp_code_splitter

def test_code_splitter(python_code_splitter, JS_code_splitter, csharp_code_splitter):
    python_passages = python_code_splitter.split_document(python_TEST_DOCUMENT)
    JS_passages = JS_code_splitter.split_document(JS_doc_TEST_DOCUMENT)
    csharp_passages = csharp_code_splitter.split_document(csharp_TEST_DOCUMENT)

    test_passages = [python_passages, JS_passages, csharp_passages]

    for passages in test_passages:
        assert len(passages) > 1
        assert passages[0].next_passage_id == passages[1].id
        assert passages[1].previous_passage_id == passages[0].id
        assert passages[0].filepath == 'test_source'
        assert passages[0].filepath == passages[1].filepath
        assert passages[0].previous_passage_id is None
        assert passages[-1].next_passage_id is None


    # Check passage split well.(Based our test document)
    ## python (language_name = 'PYTHON', chunk_size = 50, chunk_overlap = 0)
    assert len(python_passages) == 2

    ## JS (language_name = 'JS', chunk_size = 60, chunk_overlap = 0)
    assert len(JS_passages) == 2
    
    ## C# (language_name = 'CSHARP' chunk_size = 17, chunk_overlap = 0)
    assert len(csharp_passages) == 33


    # Check splitter preserve other metadata in original document.
    ## python
    test_document_metadata = list(copy.deepcopy(python_TEST_DOCUMENT).metadata.items())
    test_document_metadata.pop(0)
    for passage in python_passages:
        for element in test_document_metadata:
            assert element in list(passage.metadata_etc.items())

    ## JS
    test_document_metadata = list(copy.deepcopy(JS_doc_TEST_DOCUMENT).metadata.items())
    test_document_metadata.pop(0)
    for passage in JS_passages:
        for element in test_document_metadata:
            assert element in list(passage.metadata_etc.items())

    ## C#
    test_document_metadata = list(copy.deepcopy(csharp_TEST_DOCUMENT).metadata.items())
    test_document_metadata.pop(0)
    for passage in csharp_passages:
        for element in test_document_metadata:
            assert element in list(passage.metadata_etc.items())



    # Check passages' metadata_etc
    ## metadata_etc can't contain file path.
    for passages in test_passages:
        assert ('source', 'test_source') not in list(passages[0].metadata_etc.items())
        assert ('source', 'test_source') not in list(passages[-1].metadata_etc.items())


