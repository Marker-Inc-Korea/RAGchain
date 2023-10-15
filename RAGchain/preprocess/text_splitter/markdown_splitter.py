from typing import List
from uuid import uuid4

from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter

from RAGchain.preprocess.text_splitter.base import BaseTextSplitter
from RAGchain.schema import Passage


# 마크다운 헤더로 자르는 splitter
class MarkDownHeaderSplitter(BaseTextSplitter):
    def __init__(self, headers_to_split_on: List[tuple[str, str]], return_each_line: bool = False):
        """
        :param headers_to_split_on: A list of tuples which appended  to create split standard.
        ex)
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        """
        self.markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, return_each_line)

    def split_document(self, documents: Document):
        check_metadata = documents
        split_documents = self.markdown_splitter.split_text(documents.page_content)

        # Modify meta_data's keys and values.
        # 람다 써보기
        # 키 값을 지정해야하는데 zip 사용?
        passages = []

        doc = documents.metadata
        split = split_documents[0].metadata.copy()
        test_meta = dict(doc, **split)

        ids = [uuid4() for _ in range(len(split_documents))]
        for i, (split_document, uuid) in enumerate(zip(split_documents, ids)):
            split = split_document.metadata.copy()
            metadata_etc = dict(documents.metadata,
                                **split_document.metadata.copy())  # metadata_etc = doc's metadata_etc + headers
            filepath = documents.metadata.pop('source')  # user doc's metadata value.
            previous_passage_id = ids[i - 1] if i > 0 else None
            next_passage_id = ids[i + 1] if i < len(split_documents) - 1 else None
            passage = Passage(id=uuid,
                              content=split_document.page_content,
                              filepath=filepath,
                              previous_passage_id=previous_passage_id,
                              next_passage_id=next_passage_id,
                              metadata_etc=metadata_etc)
            passages.append(passage)
        print(f"Split into {len(passages)} passages")

        return passages

    """
    [Passage(id=UUID('9c57dfb1-13a5-47a3-bad9-14192e8c632f'), content="To fix the issue of not being able to import your module when running 'pytest', you can try the following solutions:", filepath='test_source', next_passage_id=UUID('39b51ae3-3199-4793-8e4e-a3144debcdb1')),
     Passage(id=UUID('39b51ae3-3199-4793-8e4e-a3144debcdb1'), content='Add empty __init__.py files to all subdirectories under the src/ directory. This will cause pytest to import \neverything using package/module names that start with directory names under src/. [0] Modify the PYTHONPATH \nenvironment variable to include the root directory of your project. This can be done by running the command export \nPYTHONPATH=/path/to/your/project in Linux/Unix systems. By adding the root directory to PYTHONPATH, Python will be', filepath='test_source', previous_passage_id=UUID('9c57dfb1-13a5-47a3-bad9-14192e8c632f'), next_passage_id=UUID('d148e22c-fcd9-4c18-810f-2a7fc2aa8718')), 
     Passage(id=UUID('d148e22c-fcd9-4c18-810f-2a7fc2aa8718'), content='able to find your modules from the test directory. You can then run pytest using PYTHONPATH=. pytest. [0] Use the \n--import-mode command-line flag in pytest to control how test modules are imported. The default mode is prepend, \nwhich inserts the directory path containing each module at the beginning of sys.path. You can try using the append \nmode instead, which appends the directory containing each module to the end of sys.path. This can be useful if you', filepath='test_source', previous_passage_id=UUID('39b51ae3-3199-4793-8e4e-a3144debcdb1'), next_passage_id=UUID('c0760c5a-0f7c-4115-b053-ee17657f52f0')), Passage(id=UUID('c0760c5a-0f7c-4115-b053-ee17657f52f0'), content='want to run test modules against installed versions of a package. For example, if you have a package under test and a \nseparate test package, using --import-mode=append will allow pytest to pick up the installed version of the package \nunder test. [2] Make sure that there is no __init__.py file in the folder containing your tests. Having an \n__init__.py file in the tests folder can cause import issues with pytest. If you have an __init__.py file in the', filepath='test_source', previous_passage_id=UUID('d148e22c-fcd9-4c18-810f-2a7fc2aa8718'), next_passage_id=UUID('39e4b244-eede-47bb-8f2c-ddb2548198d1')), Passage(id=UUID('39e4b244-eede-47bb-8f2c-ddb2548198d1'), content='tests folder, try removing it and see if that solves the problem. [6] [7] Run pytest using the python -m pytest \ncommand instead of just pytest. This will add the current directory to sys.path, which might resolve import issues. [\n7] Here is a summary of the steps:', filepath='test_source', previous_passage_id=UUID('c0760c5a-0f7c-4115-b053-ee17657f52f0'), 
    """
