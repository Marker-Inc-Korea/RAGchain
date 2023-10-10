import pytest
from langchain.schema import Document

from RAGchain.preprocess.text_splitter import RecursiveTextSplitter

TEST_DOCUMENT = Document(
    page_content="""
    To fix the issue of not being able to import your module when running 'pytest', you can try the following solutions:

Add empty __init__.py files to all subdirectories under the src/ directory. This will cause pytest to import 
everything using package/module names that start with directory names under src/. [0] Modify the PYTHONPATH 
environment variable to include the root directory of your project. This can be done by running the command export 
PYTHONPATH=/path/to/your/project in Linux/Unix systems. By adding the root directory to PYTHONPATH, Python will be 
able to find your modules from the test directory. You can then run pytest using PYTHONPATH=. pytest. [0] Use the 
--import-mode command-line flag in pytest to control how test modules are imported. The default mode is prepend, 
which inserts the directory path containing each module at the beginning of sys.path. You can try using the append 
mode instead, which appends the directory containing each module to the end of sys.path. This can be useful if you 
want to run test modules against installed versions of a package. For example, if you have a package under test and a 
separate test package, using --import-mode=append will allow pytest to pick up the installed version of the package 
under test. [2] Make sure that there is no __init__.py file in the folder containing your tests. Having an 
__init__.py file in the tests folder can cause import issues with pytest. If you have an __init__.py file in the 
tests folder, try removing it and see if that solves the problem. [6] [7] Run pytest using the python -m pytest 
command instead of just pytest. This will add the current directory to sys.path, which might resolve import issues. [
7] Here is a summary of the steps:

Add empty __init__.py files to all subdirectories under the src/ directory.
Modify the PYTHONPATH environment variable to include the root directory of your project.
Run pytest using PYTHONPATH=. pytest.
Use the --import-mode command-line flag in pytest to control how test modules are imported.
Make sure there is no __init__.py file in the tests folder.
Run pytest using the python -m pytest command.
These solutions should help resolve the import issues you are facing when running pytest.
""",
    metadata={
        'source': 'test_source'
    }
)


@pytest.fixture
def recursive_text_splitter():
    recursive_text_splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
    yield recursive_text_splitter


def test_recursive_text_splitter(recursive_text_splitter):
    passages = recursive_text_splitter.split_document(TEST_DOCUMENT)
    assert len(passages) > 1
    assert passages[0].next_passage_id == passages[1].id
    assert passages[1].previous_passage_id == passages[0].id
    assert passages[0].filepath == 'test_source'
    assert passages[0].filepath == passages[1].filepath
    assert passages[0].previous_passage_id is None
    assert passages[-1].next_passage_id is None
    assert TEST_DOCUMENT.page_content.strip()[:10] == passages[0].content[:10]
    assert TEST_DOCUMENT.page_content.strip()[-10:] == passages[-1].content[-10:]
