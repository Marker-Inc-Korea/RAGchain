from typing import List
import os
from transformers import StoppingCriteria
import torch


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


def slice_stop_words(input_str: str, stop_words: List[str]):
    for stop_word in stop_words:
        if stop_word in input_str:
            temp_ans = input_str.split(stop_word)[0]
            if temp_ans:
                input_str = temp_ans
    return input_str


class FileChecker:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_type = os.path.splitext(file_path)[-1].lower()

    def check_type(self, file_type: str = None, file_types: List[str] = None):
        if file_types is not None:
            checks = [self.file_type == file_type for file_type in file_types]
            if not any(checks):
                raise ValueError(
                    f"FileChecker.check_file_type: file type must be one of file types, but got {self.file_type}")
            return self
        if self.file_type != file_type:
            raise ValueError(f"FileChecker.check_file_type: file type must be {file_type}, but got {self.file_type}")
        return self

    def is_exist(self):
        return os.path.exists(self.file_path)

    def __str__(self):
        return self.file_path
