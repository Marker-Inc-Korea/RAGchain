import os
import pytest

from RAGchain.preprocess.loader.hwp_table_loader import HwpLoader

file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir, "resources",
                         "ingest_files", "hwp_sample.hwp")
table_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, os.pardir,
                               "resources", "ingest_files", "hwp_table_sample.hwp")


def test_hwp_table_loader_doc():
    docs = HwpLoader(file_path).load()

    assert len(docs) != 0
    assert os.path.exists(file_path + "x") == 0


def test_hwp_table_loader_table():
    tables = HwpLoader(table_file_path).load_table()

    assert len(tables) != 0
    assert os.path.exists(table_file_path + "x") == 0
