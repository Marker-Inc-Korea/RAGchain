import os
import pathlib

import pytest

from RAGchain.preprocess.loader.mathpix_markdown_loader import MathpixMarkdownLoader

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
file_dir = os.path.join(root_dir, "resources", "ingest_files")
mmd_filepath = os.path.join(file_dir, "test.mmd")


@pytest.fixture
def mathpix_markdown_loader():
    assert os.path.exists(mmd_filepath)
    yield MathpixMarkdownLoader(mmd_filepath)


def test_mathpix_markdown_loader(mathpix_markdown_loader):
    # test no split section & table
    docs = mathpix_markdown_loader.load(split_section=False, split_table=False)
    assert len(docs) == 1
    assert bool(docs[0].page_content)

    # test only split section
    docs = mathpix_markdown_loader.load(split_section=True, split_table=False)
    assert len(docs) == 11
    for doc in docs:
        assert bool(doc.page_content)
        assert doc.metadata["content_type"] == "text"

    # test only split table
    docs = mathpix_markdown_loader.load(split_section=False, split_table=True)
    assert len(docs) == 7
    for i, doc in enumerate(docs):
        assert bool(doc.page_content)
        if i % 2 == 0:
            assert doc.metadata["content_type"] == "text"
        else:
            assert doc.metadata["content_type"] == "table"

    # test split section & table
    docs = mathpix_markdown_loader.load(split_section=True, split_table=True)
    assert len(docs) == 17
    for doc in docs:
        assert bool(doc.page_content)
        if doc.metadata["content_type"] == "table":
            assert doc.page_content.startswith("\\\\begin{table}")
        else:
            assert bool(doc.page_content)


def test_split_section():
    sample_txt = """## Abstract
    Hi! I'm a abstract.
    
    ### Introduction
    Hi! I'm a introduction.
    
    ### Related Work
    Hi! I'm a related work.
    """

    split_texts = MathpixMarkdownLoader.split_section(sample_txt)
    assert len(split_texts) == 3
    assert split_texts[0].strip() == "## Abstract\n    Hi! I'm a abstract."
    assert split_texts[1].strip() == "### Introduction\n    Hi! I'm a introduction."
    assert split_texts[2].strip() == "### Related Work\n    Hi! I'm a related work."


def test_split_table():
    sample_txt = """## 4. Our Method: BM25
\\\\begin{table}
\\\\begin{tabular} 34 \\\\ 43 \\\\ 34 \\\\ 43 \\\\end{tabular}
\\\\end{table}
Table 1. COLIEE 2021 task 1 data statistics.
    """
    split_texts = MathpixMarkdownLoader.split_table(sample_txt)
    assert len(split_texts) == 3
    assert split_texts[0].strip() == "## 4. Our Method: BM25"
    assert split_texts[
               1].strip() == "\\\\begin{table}\n\\\\begin{tabular} 34 \\\\ 43 \\\\ 34 \\\\ 43 \\\\end{tabular}\n\\\\end{table}"
    assert split_texts[2].strip() == "Table 1. COLIEE 2021 task 1 data statistics."
