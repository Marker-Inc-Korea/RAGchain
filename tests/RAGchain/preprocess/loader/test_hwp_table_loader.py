import os
import pytest

from RAGchain.preprocess.loader.hwp_table_loader import HwpLoader

file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir, "resources", "ingest_files", "hwp_sample.hwp")
table_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir, "resources", "ingest_files", "hwp_table_sample.hwp")


def test_hwp_table_loader():
    docs = HwpLoader(file_path).load()
    tables = HwpLoader(table_file_path).load_table()

    print(docs[0].page_content)
    assert len(docs) == 1
    assert bool(docs[0].page_content)


