import pytest
from langchain.llms.openai import OpenAI

from RAGchain.pipeline import GoogleSearchRunPipeline


@pytest.fixture
def google_search_run_pipeline():
    llm = OpenAI(model_name="babbage-002")
    pipeline = GoogleSearchRunPipeline(llm)
    yield pipeline


def test_google_search_run_pipeline(google_search_run_pipeline):
    answer = google_search_run_pipeline.run.invoke("What is the capital of France?")
    assert bool(answer)

    answers = google_search_run_pipeline.run.batch(["What is the capital of France?",
                                                    "What is the capital of Germany?"])
    assert len(answers) == 2

    answers, passages, scores = google_search_run_pipeline.get_passages_and_run(["What is the capital of France?",
                                                                                 "What is the capital of Germany?"],
                                                                                top_k=2)
    assert len(answers) == len(passages) == len(scores) == 2
    assert len(passages[0]) == len(scores[0]) == 2
