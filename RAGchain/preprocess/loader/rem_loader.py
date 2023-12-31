from datetime import datetime
from typing import List, Iterator, Optional

import pytz
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document

from RAGchain.utils.util import FileChecker


class RemLoader(BaseLoader):
    """
    Load rem storage file from rem sqlite database.
    You can set time range to load.
    """

    def __init__(self, path: str, time_range: Optional[List[datetime]] = None):
        """
        :param path: rem sqlite database file path
        :param time_range: time range to load. If None, load all data. We recommend set time range.
        It will be slow when you try to load all data from once. Default is None.
        """
        self.path = path
        if not FileChecker(self.path).check_type(file_type='.sqlite3').is_exist():
            raise ValueError(f"{self.path} is not sqlite3 file or do not exist.")
        import sqlite3
        self.conn = sqlite3.connect(path)
        self.time_range = time_range if time_range is not None else [datetime(1970, 1, 1), datetime.now()]
        self.__preprocess_time_range()
        assert len(self.time_range) == 2, "time_range must be list of datetime with length 2"

    def lazy_load(self) -> Iterator[Document]:
        query = f"""
                    SELECT allText.text, frames.timestamp
                    FROM allText
                    JOIN frames ON allText.frameId = frames.id
                    WHERE frames.timestamp BETWEEN '{self.time_range[0]}' AND '{self.time_range[1]}'
                """
        cur = self.conn.cursor()
        cur.execute(query)
        for row in cur.fetchall():
            yield Document(page_content=row[0],
                           metadata={
                               "source": self.path,
                               "content_datetime": datetime.strptime(row[1], '%Y-%m-%dT%H:%M:%S.%f'),
                           })

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def __preprocess_time_range(self):
        for i, time in enumerate(self.time_range):
            alter_time = time.astimezone(pytz.UTC)
            self.time_range[i] = alter_time.strftime('%Y-%m-%dT%H:%M:%S.%f')
