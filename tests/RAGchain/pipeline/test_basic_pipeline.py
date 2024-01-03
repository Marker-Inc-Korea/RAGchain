import logging
import os
import pathlib

import pytest
from langchain.llms.openai import OpenAI
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable.history import RunnableWithMessageHistory

from RAGchain.DB import MongoDB
from RAGchain.pipeline.basic import BasicIngestPipeline, BasicRunPipeline
from RAGchain.preprocess.loader import FileLoader
from RAGchain.retrieval import BM25Retrieval
from RAGchain.schema.prompt import RAGchainChatPromptTemplate

log = logging.getLogger(__name__)

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent
file_dir = os.path.join(root_dir, "resources", "ingest_files")
bm25_path = os.path.join(root_dir, "resources", "bm25", "bm25_basic_pipeline.pkl")
mongodb_collection_name = "test_basic_pipeline"
mongodb_config = {
    "mongo_url": os.getenv('MONGO_URL'),
    "db_name": os.getenv('MONGO_DB_NAME'),
    "collection_name": mongodb_collection_name
}


@pytest.fixture
def basic_run_pipeline():
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    ingest_pipeline = BasicIngestPipeline(
        file_loader=FileLoader(file_dir, os.getenv('HWP_CONVERTER_HOST')),
        db=MongoDB(**mongodb_config),
        retrieval=BM25Retrieval(bm25_path)
    )
    ingest_pipeline.run.invoke(None)

    pipeline = BasicRunPipeline(
        retrieval=BM25Retrieval(bm25_path),
        llm=OpenAI()
    )
    yield pipeline
    teardown_all(mongodb_config, bm25_path)


@pytest.fixture
def basic_run_pipeline_chat_history():
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    ingest_pipeline = BasicIngestPipeline(
        file_loader=FileLoader(file_dir, os.getenv('HWP_CONVERTER_HOST')),
        db=MongoDB(**mongodb_config),
        retrieval=BM25Retrieval(bm25_path)
    )
    ingest_pipeline.run.invoke(None)
    chat_history_prompt = RAGchainChatPromptTemplate.from_messages([
        ("system", "Answer user's question based on given passages."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "Passages: {passages}"),
        ("ai", "What is your question? I will answer based on given passages."),
        ("user", "Question: {question}"),
        ("ai", "Answer:")
    ])
    pipeline = BasicRunPipeline(
        retrieval=BM25Retrieval(bm25_path),
        llm=OpenAI(),
        prompt=chat_history_prompt
    )
    yield pipeline
    teardown_all(mongodb_config, bm25_path)


def test_basic_pipeline(basic_run_pipeline):
    assert os.path.exists(bm25_path)
    query = "What is the purpose of RAGchain project? And what inspired it?"
    log.info(f"query: {query}")
    answer = basic_run_pipeline.run.invoke(query)
    assert bool(answer) is True
    log.info(f"answer: {answer}")

    queries = ["What is the purpose of KoPrivateGPT project?",
               "What inspired KoPrivateGPT project?",
               "How can I install KoPrivateGPT project?"]
    answers, passages, scores = basic_run_pipeline.get_passages_and_run(queries, top_k=4)
    assert len(answers) == len(queries)
    assert len(passages) == len(queries)
    assert len(scores) == len(queries)
    for query, answer, passage, score in zip(queries, answers, passages, scores):
        assert bool(answer) is True
        log.info(f"question: {query}\nanswer: {answer}")
        assert len(passage) == 4
        log.info(f"score: {score}")


def test_chat_history(basic_run_pipeline_chat_history):
    chat_history = ChatMessageHistory()
    chain_with_history = RunnableWithMessageHistory(
        basic_run_pipeline_chat_history.run,
        lambda session_id: chat_history,
        input_messages_key="question",
        history_messages_key="history"
    )

    answer = chain_with_history.invoke({"question": "How can I install this project?"},
                                       config={"configurable": {"session_id": "test_session"}})
    assert bool(answer)
    log.info(f"answer: {answer}")

    answer = chain_with_history.invoke({"question": "Is there other things to do?"},
                                       config={"configurable": {"session_id": "test_session"}})
    assert bool(answer)
    log.info(f"answer: {answer}")


def teardown_all(mongo_config, path):
    # teardown mongo db
    mongo_db = MongoDB(**mongo_config)
    mongo_db.create_or_load()
    assert mongo_db.collection_name == mongodb_collection_name
    mongo_db.collection.drop()
    assert mongodb_collection_name not in mongo_db.db.list_collection_names()

    # teardown bm25
    if os.path.exists(path):
        os.remove(path)
