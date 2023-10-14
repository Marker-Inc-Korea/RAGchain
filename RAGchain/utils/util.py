import os
from typing import List, Optional

import openai


def slice_stop_words(input_str: str, stop_words: List[str]):
    for stop_word in stop_words:
        if stop_word in input_str:
            temp_ans = input_str.split(stop_word)[0]
            if temp_ans:
                input_str = temp_ans
    return input_str


def set_api_base(api_base: str):
    if api_base is None:
        if os.getenv("OPENAI_API_KEY") is None:
            raise ValueError("Please set OPENAI_API_KEY as environment variable")
        openai.api_key = os.environ["OPENAI_API_KEY"]
    else:
        openai.api_base = api_base


def text_modifier(text: str, modify_words: Optional[List[str]] = None) -> List[str]:
    """
    You have to separate each word with underbar '_'
    """
    result = [text, text.lower(), text.capitalize(), text.upper()]
    if "_" in text:
        text_list = text.split("_")
        result.append("-".join(text_list))
        result.append("_".join([text.capitalize() for text in text_list]))
        result.append("-".join([text.capitalize() for text in text_list]))
        result.append("".join(text_list))
        result.append("".join([text.capitalize() for text in text_list]))
        result.append("".join([text.upper() for text in text_list]))
    if modify_words is not None:
        result.append(modify_words)
    return result


class FileChecker:
    """
    FileChecker is a class to check file type and existence.
    """

    def __init__(self, file_path: str):
        """
        :param file_path: str, file path to check.
        """
        self.file_path = file_path
        self.file_type = os.path.splitext(file_path)[-1].lower()

    def check_type(self, file_type: str = None, file_types: List[str] = None):
        """
        :param file_type: str, file type to check. Default is None. You must use this when you want to check only one file type.
        When you use this, you don't need to use file_types.
        :param file_types: List[str], file types to check. Default is None. You must use this when you want to check multiple file types.
        When you use this, you don't need to use file_type.
        """
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
        """
        check file existence.
        :return: bool, True if file exists, else False.
        """
        return os.path.exists(self.file_path)

    def __str__(self):
        return self.file_path
