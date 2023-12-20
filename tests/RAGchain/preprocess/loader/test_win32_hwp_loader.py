import logging
import os
import pathlib

from RAGchain.preprocess.loader import Win32HwpLoader

logger = logging.getLogger(__name__)

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
file_path = os.path.join(root_dir, "resources", "ingest_files", "test.hwp")
table_file_path = os.path.join(root_dir, "resources", "ingest_files", "hwp_table_sample.hwp")


def test_win32_hwp_loader_doc():
    docs = Win32HwpLoader(file_path).load()

    assert len(docs) == 1
    assert bool(docs[0].page_content)
    logger.info(f'loaded file content : {docs[0].page_content}')


def test_win32_hwp_loader_table():
    tables = Win32HwpLoader(table_file_path).load()

    assert len(tables) == 3
    assert bool(tables[0].page_content)
    assert tables[0].metadata['page_type'] == 'text'
    assert tables[0].metadata['source'] == table_file_path

    assert tables[1].metadata['page_type'] == 'table'
