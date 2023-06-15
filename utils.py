import openpyxl
import csv
import tempfile
from transformers import StoppingCriteria
import torch

def xlxs_to_csv(file_path: str, sheet_name: str = None) -> list[str]:
    """
    Convert a workbook into a list of csv files
    :param file_path: the path to the workbook
    :param sheet_name: the name of the sheet to convert
    :return: a list of temporary file names
    """
    # Load the workbook and select the active worksheet
    wb = openpyxl.load_workbook(file_path)
    # ws = wb.active
    #
    # # Create a new temporary file and write the contents of the worksheet to it
    # with tempfile.NamedTemporaryFile(mode='w+', newline='', delete=False) as f:
    #     c = csv.writer(f)
    #     for r in ws.rows:
    #         c.writerow([cell.value for cell in r])
    # return f.name
    # load all sheets if sheet_name is None
    wb = wb if sheet_name is None else [wb[sheet_name]]
    temp_file_name = []
    # Iterate over the worksheets in the workbook
    for ws in wb:
        # Create a new temporary file and write the contents of the worksheet to it
        with tempfile.NamedTemporaryFile(mode='w+', newline='', suffix='.csv', delete=False) as f:
            c = csv.writer(f)
            for r in ws.rows:
                c.writerow([cell.value for cell in r])
            temp_file_name.append(f.name)
    # print(f'all Sheets are saved to temporary file {temp_file_name}')
    return temp_file_name


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
      super().__init__()
      self.stops = stops
      self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
      stop_count = 0
      for stop in self.stops:
        stop_count = (stop == input_ids[0]).sum().item()

      if stop_count >= self.ENCOUNTERS:
          return True
      return False
