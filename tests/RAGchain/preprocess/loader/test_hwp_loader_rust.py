import os
import pytest

from RAGchain.preprocess.loader.hwp_loader_rust import HwpLoaderRust

file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir, "resources",
                         "ingest_files", "Test.hwp")
table_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir,
                               "resources", "ingest_files", "hwp_table_sample.hwp")


def test_hwp_table_loader_doc():
    docs = HwpLoaderRust(file_path).load()

    assert len(docs) != 0


def test_hwp_table_loader_table():
    tables = HwpLoaderRust(table_file_path).load_table()

    assert len(tables) != 0
