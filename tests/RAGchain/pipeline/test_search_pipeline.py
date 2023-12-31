import pytest

from RAGchain.pipeline import GoogleSearchRunPipeline

from langchain.llms.openai import OpenAI


@pytest.fixture
def google_search_run_pipeline():
    llm = OpenAI(model_name="babbage-002")
    pipeline = GoogleSearchRunPipeline(llm)
    yield pipeline


def test_google_search_run_pipeline(google_search_run_pipeline):
    answer = google_search_run_pipeline.run.invoke({"question": "What is the capital of France?"})
    assert bool(answer)

    answers, passages, scores = google_search_run_pipeline.get_passages_and_run(["What is the capital of France?",
                                                                                   "What is the capital of Germany?"])
    assert len(answers) == len(passages) == len(scores) == 2
    assert len(passages[0]) == len(scores[0]) == 3
