import os
import re
import zipfile
import shutil
from typing import List

import win32com.client as win32
from bs4 import BeautifulSoup
from langchain.docstore.document import Document

from langchain.document_loaders.base import BaseLoader


class HwpLoader(BaseLoader):
    def __init__(self, path: str, *args, **kwargs):
        self.file_path = path
        self.result = []
        self.flag = 0

    def convert_hwp_to_hwpx(self):
        hwp = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
        hwp.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
        hwp.HParameterSet.HTableCreation.TableProperties.TreatAsChar = 1
        hwp.Open(self.file_path)
        hwp.SaveAs(hwp.Path + "X", "HWPX")
        hwp.Quit()
        self.file_path = self.file_path + "x"

    def unzip_hwpx(self, file_path):
        os.chdir(os.path.dirname(self.file_path))
        target_path = os.path.join(os.getcwd(), "hwpx")
        with zipfile.ZipFile(self.file_path, 'r') as zf:
            zf.extractall(path=target_path)
        # os.remove(self.file_path)

    def spliter(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            xml_content = file.read()

        separate = re.split(r'<hp:tbl|</hp:tbl>', xml_content)
        return separate

    def xml_to_html(self, xml):
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

    def preprocessor(self):

        if self.file_path[-5:] == ".hwpx":
            self.unzip_hwpx(self.file_path)

        elif self.file_path[-4:] == ".hwp":

            self.convert_hwp_to_hwpx()
            self.unzip_hwpx(self.file_path)

        else:
            raise ValueError("The file extension must be .hwp or .hwpx")

        #print("Loading {0}".format(self.file_path))

        pattern = r'</?(?!(?:em|strong)\b)[a-z](?:[^>"\']|"[^"]*"|\'[^\']*\')*>'

        for i, xml in enumerate(self.spliter(os.path.join(os.getcwd(), "hwpx", "Contents", "section0.xml"))):
            if i % 2 == 0:
                self.result.append(re.sub(pattern, '', xml))
            if i % 2 == 1:
                self.result.append('<hp:tbl' + xml)

        self.result[0] = self.result[0].strip("""['<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>""")

        for i, xml in enumerate(self.result):
            if i % 2 == 1:
                self.result[i] = self.xml_to_html(xml)

        #print("Preprocessing is done")
        return self.result

    def load(self) -> List[Document]:
        document_list = []
        page = ""
        if self.flag == 0:
            self.preprocessor()
            self.flag = 1
            return self.load()

        elif self.flag == 1:
            for i, txt in enumerate(self.result):
                if i % 2 == 0:
                    page += txt
            print(page)
            document_list = [Document(page_content=page, metadata={"source": self.file_path})]

        if os.path.exists(os.path.join(self.file_path, os.pardir, "hwpx")):
            shutil.rmtree(os.path.join(self.file_path, os.pardir, "hwpx"))
            os.remove(os.path.join(self.file_path))
        return document_list

    def load_table(self) -> List[Document]:
        document_list = []

        if self.flag == 0:
            self.preprocessor()
            self.flag = 1
            return self.load_table()

        elif self.flag == 1:
            document_list = []
            for i, xml_table in enumerate(self.result):
                if i % 2 == 1:
                    document_list.append(Document(page_content=xml_table, metadata={"source": self.file_path}))

        if os.path.exists(os.path.join(self.file_path, os.pardir, "hwpx")):
            shutil.rmtree(os.path.join(self.file_path, os.pardir, "hwpx"))
            os.remove(os.path.join(self.file_path))
        return document_list
