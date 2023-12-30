import pytest

from RAGchain.utils.websearch import GoogleSearch


def test_google_search():
    search = GoogleSearch()
    passages = search.get_search_data("뉴진스 민지의 생일은?", num_results=2)
    assert len(passages) == 2
