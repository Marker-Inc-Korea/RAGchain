import os
import re
import tempfile
import zipfile
from typing import List, Iterator

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


class Win32HwpLoader(BaseLoader):
    """
    Load HWP file using pywin32. It works for only Windows.
    Using load or lazy_load, you can get list of Documents from hwp file.
    This loader loads all paragraphs and tables from hwp file.
    At the first Document, there are all paragraphs from hwp file, excluding texts in each table.
    Next, there are separated Documents for each table. All table contents are converted to html format.
    So you can get row, columns, or any complicated table structure.

    In the metadata, there are filepath at key 'source' and page_type, which is 'text' or 'table'.

    It is great option to use loader for loading complicated tables from hwp file.
    But it is only available at windows, so choose other hwp loader if you want to use at mac or linux.
    """
    def __init__(self, path: str):
        """
        :param path: hwp file path
        """
        self.file_path = path

    def lazy_load(self) -> Iterator[Document]:
        text, tables = self.preprocessor()

        yield Document(page_content=" ".join(text), metadata={"source": self.file_path,
                                                              'page_type': 'text'})
        for table in tables:
            yield Document(page_content=table, metadata={"source": self.file_path, 'page_type': 'table'})

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def preprocessor(self) -> tuple[List, List]:
        text = list()
        table = list()

        hwpx_temp_file = None
        if self.file_path.endswith('.hwp'):
            hwpx_temp_file = tempfile.NamedTemporaryFile(suffix='.hwpx', mode='w', delete=False)
            self.convert_hwp_to_hwpx(self.file_path, hwpx_temp_file.name)
            hwpx_file = hwpx_temp_file.name
        elif self.file_path.endswith('.hwpx'):
            hwpx_file = self.file_path
        else:
            raise ValueError("The file extension must be .hwp or .hwpx")

        with tempfile.TemporaryDirectory() as target_path:
            with zipfile.ZipFile(hwpx_file, 'r') as zf:
                zf.extractall(path=target_path)

            if hwpx_temp_file is not None:
                hwpx_temp_file.close()
                os.unlink(hwpx_temp_file.name)

            text_extract_pattern = r'</?(?!(?:em|strong)\b)[a-z](?:[^>"\']|"[^"]*"|\'[^\']*\')*>'

            for i, xml in enumerate(self.__splitter(os.path.join(target_path, "Contents", "section0.xml"))):
                if i % 2 == 0:
                    text.append(re.sub(text_extract_pattern, '', xml))  # just text
                elif i % 2 == 1:
                    table.append('<hp:tbl' + xml)  # table

            text[0] = text[0].strip("""['<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>""")
            table = list(map(self.__xml_to_html, table))

            return text, table

    @staticmethod
    def convert_hwp_to_hwpx(input_filepath, output_filepath):
        try:
            import win32com.client as win32
        except ImportError:
            raise ImportError("Please install pywin32."
                              "pip install pywin32")

        hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
        hwp.HParameterSet.HTableCreation.TableProperties.TreatAsChar = 1
        hwp.Open(input_filepath)
        hwp.SaveAs(output_filepath, "HWPX")
        hwp.Quit()

    @staticmethod
    def __splitter(path):
        with open(path, 'r', encoding='utf-8') as file:
            xml_content = file.read()

        separate = re.split(r'<hp:tbl|</hp:tbl>', xml_content)
        return separate

    @staticmethod
    def __xml_to_html(xml):
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("Please install bs4."
                              "pip install bs4")
        bs = BeautifulSoup(xml, 'html.parser')
        flag, line = 0, 0
        result_txt = """<table
                    border="1"
                    width="50%"
                    height="200"
                    cellspacing="5">\n"""
        for tag in bs.find_all('hp:tr'):
            result_txt += "\t <tr>\n"
            for tag2 in tag.find_all('hp:tc'):
                for tag3 in tag2.find_all('hp:cellspan'):
                    for tag4 in tag2.find_all('hp:sublist'):
                        result_txt += '\t \t <td>'
                        for tag5 in tag4.find_all('hp:t'):
                            if tag3.attrs['colspan'] != "1" and tag3.attrs['rowspan'] == "1" and flag == 0:
                                result_txt = result_txt[:-1] + ' colspan ="{}">'.format(
                                    tag3.attrs['colspan']) + tag5.get_text()
                                flag = 1
                            elif tag3.attrs['colspan'] == "1" and tag3.attrs[
                                'rowspan'] != "1" and flag == 0 and line == 0:
                                result_txt = result_txt[:-1] + ' rowspan ="{}">'.format(
                                    tag3.attrs['rowspan']) + tag5.get_text()
                                flag = 1
                                line = 2
                            elif tag3.attrs['colspan'] != "1" and tag3.attrs['rowspan'] != "1" and flag == 0:
                                result_txt = result_txt[:-1] + ' colspan ="{}" rowspan ="{}">'.format(
                                    tag3.attrs['colspan'], tag3.attrs['rowspan']) + tag5.get_text()
                                flag = 1
                            else:
                                result_txt += tag5.get_text()
                            flag = 0
                        result_txt += '</td>\n'
                        line = 0
            result_txt += '\t </tr>\n'
        result_txt += '</table>'
        return result_txt
