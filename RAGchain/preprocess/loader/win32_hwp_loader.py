import os
import re
import zipfile
from typing import List, Iterator

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


class Win32HwpLoader(BaseLoader):
    def __init__(self, path: str):
        self.file_path = path
        self.hwp_file_path = self.file_path
        self.result = []
        self.flag = 0

    def convert_hwp_to_hwpx(self):
        try:
            import win32com.client as win32
        except ImportError:
            raise ImportError("Please install pywin32."
                              "pip install pywin32")
        # TODO: make this tempFile
        hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
        hwp.HParameterSet.HTableCreation.TableProperties.TreatAsChar = 1
        hwp.Open(self.file_path)
        hwp.SaveAs(hwp.Path + "x", "HWPX")
        hwp.Quit()
        self.hwp_file_path = self.file_path + "x"

    def unzip_hwpx(self, file_path):
        # TODO: make this tempFile
        os.chdir(os.path.dirname(file_path))
        target_path = os.path.join(os.getcwd(), "hwpx")
        with zipfile.ZipFile(file_path, 'r') as zf:
            zf.extractall(path=target_path)
        # os.remove(self.file_path)

    def splitter(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            xml_content = file.read()

        separate = re.split(r'<hp:tbl|</hp:tbl>', xml_content)
        return separate

    def xml_to_html(self, xml):
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

    def preprocessor(self) -> tuple[List, List]:
        text = list()
        table = list()

        if self.file_path.endswith(".hwpx"):
            self.unzip_hwpx(self.hwp_file_path)
        elif self.file_path.endswith(".hwp"):
            self.convert_hwp_to_hwpx()
            self.unzip_hwpx(self.hwp_file_path)
        else:
            raise ValueError("The file extension must be .hwp or .hwpx")

        text_extract_pattern = r'</?(?!(?:em|strong)\b)[a-z](?:[^>"\']|"[^"]*"|\'[^\']*\')*>'

        for i, xml in enumerate(self.splitter(os.path.join(os.getcwd(), "hwpx", "Contents", "section0.xml"))):
            if i % 2 == 0:
                text.append(re.sub(text_extract_pattern, '', xml))  # just text
            elif i % 2 == 1:
                table.append('<hp:tbl' + xml)  # table

        text[0] = text[0].strip("""['<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>""")
        table = list(map(self.xml_to_html, table))

        return text, table

    def lazy_load(self) -> Iterator[Document]:
        text, tables = self.preprocessor()

        yield Document(page_content=" ".join(text), metadata={"source": self.file_path,
                                                              'page_type': 'text'})
        for table in tables:
            yield Document(page_content=table, metadata={"source": self.file_path, 'page_type': 'table'})

    def load(self) -> List[Document]:
        return list(self.lazy_load())
