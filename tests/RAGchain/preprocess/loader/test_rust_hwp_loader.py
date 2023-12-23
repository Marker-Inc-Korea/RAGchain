import os
import pathlib

from RAGchain.preprocess.loader import RustHwpLoader

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
file_path = os.path.join(root_dir, "resources", "ingest_files", "test.hwp")
table_file_path = os.path.join(root_dir, "resources", "ingest_files", "hwp_table_sample.hwp")


def test_rust_hwp_loader_doc():
    docs = RustHwpLoader(file_path).load()

    assert len(docs) != 0
    assert bool(docs[0].page_content)
    assert docs[0].metadata['page_type'] == 'text'
    assert docs[0].metadata['source'] == file_path


def test_rust_hwp_loader_table():
    docs = RustHwpLoader(table_file_path).load()
    assert len(docs) == 3
    assert bool(docs[0].page_content)
    assert docs[0].metadata['page_type'] == 'text'
    assert docs[0].metadata['source'] == table_file_path

    assert bool(docs[1].page_content)
    assert docs[1].metadata['page_type'] == 'table'
    assert docs[1].metadata['source'] == table_file_path

    assert docs[2].metadata['page_type'] == 'table'
