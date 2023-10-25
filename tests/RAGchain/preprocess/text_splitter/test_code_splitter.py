import copy
import pytest
from langchain.schema import Document

from RAGchain.preprocess.text_splitter import CodeSplitter

# Which file do you want to test ?
# PYTHON test
Python_doc = Document(
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
TEST_DOCUMENT = Csharp_doc


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


    # Check passage split well.(Based our test document)
    ## python (language_name = 'PYTHON', chunk_size = 50, chunk_overlap = 0)
    assert len(passages) == 2
    """
    ## JS (language_name = 'JS', chunk_size = 60, chunk_overlap = 0)
    assert len(passages) == 2
    
    ## C# (language_name = 'CSHARP' chunk_size = 17, chunk_overlap = 0)
    assert len(passages) == 33
    
    # Check splitter preserve other metadata in original document.
    test_document_metadata = list(copy.deepcopy(TEST_DOCUMENT).metadata.items())
    test_document_metadata.pop(0)
    for passage in passages:
        for element in test_document_metadata:
            assert element in list(passage.metadata_etc.items())
    """

    # Check passages' metadata_etc
    ## metadata_etc can't contain file path.
    assert ('source', 'test_source') not in list(passages[0].metadata_etc.items())
    assert ('source', 'test_source') not in list(passages[-1].metadata_etc.items())


