import pytest

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper


def test_google_search():
    search = GoogleSearchAPIWrapper()
    tool = Tool(
        name="GoogleSearch",
        description="Search Google for recent results",
        func=search.results("뉴진스 민지 생일은?", num_results=5),
    )
    results = tool.run()
    assert len(results) == 5

