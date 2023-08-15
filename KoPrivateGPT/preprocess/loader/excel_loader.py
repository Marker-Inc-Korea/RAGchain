import tempfile
from typing import List, Optional
import csv

from langchain.document_loaders import CSVLoader
from langchain.schema import Document
import openpyxl

from KoPrivateGPT.preprocess.loader.base import BaseLoader


class ExcelLoader(BaseLoader):
    def __init__(self, path: str, sheet_name: Optional[str] = None, *args, **kwargs):
        self.path = path
        wb = openpyxl.load_workbook(self.path)
        # load all sheets if sheet_name is None
        self.wb = wb if sheet_name is None else [wb[sheet_name]]

    def load(self) -> List[Document]:
        csv_filepaths = self.__xlxs_to_csv()
        docs = []
        for filepath, sheet_name in zip(csv_filepaths, self.wb.sheetnames):
            temp_loader = CSVLoader(filepath)
            document = temp_loader.load()[0]
            document.metadata['source'] = self.path
            document.metadata['sheet_name'] = sheet_name
            docs.append(document)
        return docs

    def __xlxs_to_csv(self) -> List[str]:
        temp_file_name = []
        # Iterate over the worksheets in the workbook
        for ws in self.wb:
            # Create a new temporary file and write the contents of the worksheet to it
            with tempfile.NamedTemporaryFile(mode='w+', newline='', suffix='.csv', delete=False) as f:
                c = csv.writer(f)
                for r in ws.rows:
                    c.writerow([cell.value for cell in r])
                temp_file_name.append(f.name)
        # all Sheets are saved to temporary file {temp_file_name}
        return temp_file_name
