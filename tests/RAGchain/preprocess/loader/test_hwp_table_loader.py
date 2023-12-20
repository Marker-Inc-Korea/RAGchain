import os
import pathlib

from RAGchain.preprocess.loader.win32_hwp_loader import Win32HwpLoader

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
file_path = os.path.join(root_dir, "resources", "ingest_files", "test.hwp")
table_file_path = os.path.join(root_dir, "resources", "ingest_files", "hwp_table_sample.hwp")


def test_hwp_table_loader_doc():
    docs = Win32HwpLoader(file_path).load()

    print(docs[0].page_content)


def test_hwp_table_loader_table():
    tables = Win32HwpLoader(table_file_path).load_table()

    print(tables[0].page_content)
