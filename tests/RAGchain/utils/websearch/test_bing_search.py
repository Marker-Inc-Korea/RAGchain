from langchain_core.runnables import RunnableLambda

from RAGchain.utils.websearch import BingSearch


def test_bing_search():
    search = BingSearch()
    passages = search.get_search_data("뉴진스 민지의 생일은?", num_results=2)
    assert len(passages) == 2


def test_bing_search_runnable():
    search = BingSearch()
    runnable = search | RunnableLambda(lambda x: x.to_dict())
    result = runnable.invoke(("뉴진스 민지의 생일은?", 2))
    assert isinstance(result['query'], str)
    assert result['query'] == "뉴진스 민지의 생일은?"
    assert isinstance(result['passages'], list)
    assert len(result['passages']) == 2
    assert isinstance(result['scores'], list)
    assert result['scores'] == [1.0, 0.5]
