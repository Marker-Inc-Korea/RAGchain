import os
from typing import List

import chromadb
import click
import pinecone
from dotenv import load_dotenv
from langchain.vectorstores import Chroma, Pinecone

from RAGchain.llm.basic import BasicLLM
from RAGchain.pipeline import BasicRunPipeline
from RAGchain.retrieval import BM25Retrieval, VectorDBRetrieval
from RAGchain.schema import Passage
from RAGchain.utils.embed import EmbeddingFactory
from RAGchain.utils.util import text_modifier
from config import ChromaOptions, PineconeOptions
from config import Options


def print_query_answer(query, answer):
    # Print the result
    print("\n\n> 질문:")
    print(query)
    print("\n> 대답:")
    print(answer)


def print_docs(docs: List[Passage]):
    # Print the relevant sources used for the answer
    print("----------------------------------참조한 문서---------------------------")
    for document in docs:
        print("\n> " + document.filepath + ":")
        print(document.content)
    print("----------------------------------참조한 문서---------------------------")


def select_vectordb(vectordb_type: str, embedding_type: str, device_type: str):
    load_dotenv()
    embedding_func = EmbeddingFactory(embed_type=embedding_type, device_type=device_type).get()
    chroma = Chroma(
        client=chromadb.PersistentClient(path=ChromaOptions.persist_dir),
        collection_name=ChromaOptions.collection_name,
        embedding_function=embedding_func)
    if vectordb_type in text_modifier('chroma'):
        vectordb = chroma
    elif vectordb_type in text_modifier('pinecone'):
        pinecone.init(api_key=os.getenv('PINECONE_API_KEY'), environment=os.getenv('PINECONE_ENV'))
        if PineconeOptions.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=PineconeOptions.index_name,
                metric='cosine',
                dimension=PineconeOptions.dimension
            )
        pinecone_instance = Pinecone(
            index=PineconeOptions.index_name,
            namespace=PineconeOptions.namespace,
            embedding=embedding_func,
            text_key="text"
        )
        vectordb = pinecone_instance
    else:
        raise ValueError("vectordb type is not valid")
    return vectordb


@click.command()
@click.option('--device_type', default='mps', help='device to run on, select gpu, cpu or mps')
@click.option('--retrieval_type', default='bm25', help='retrieval type to use, select vectordb or bm25')
@click.option('--vectordb_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='ko_sroberta_multitask',
              help='embedding model to use, select OpenAI or KoSimCSE.')
@click.option('--model_name', default='gpt-3.5-turbo', help='model name to use.')
@click.option('--api_base', default=None, help='api base to use.')
def main(device_type, retrieval_type: str, vectordb_type, embedding_type, model_name, api_base):
    vectordb = select_vectordb(vectordb_type, embedding_type, device_type)
    if retrieval_type in text_modifier('bm25'):
        retrieval = BM25Retrieval(save_path=Options.bm25_db_dir)
    elif retrieval_type in text_modifier('vectordb'):
        retrieval = VectorDBRetrieval(vectordb=vectordb)
    else:
        raise ValueError("retrieval type is not valid")
    pipeline = BasicRunPipeline(
        retrieval=retrieval,
        llm=BasicLLM(retrieval, model_name=model_name, api_base=api_base)
    )
    while True:
        query = input("질문을 입력하세요: ")
        if query in ["exit", "종료"]:
            break
        elif query in ["clear", "초기화"]:
            pipeline.llm.clear_chat_history()
            continue
        answer, passages = pipeline.run(query)
        print_query_answer(query, answer)
        print_docs(passages)


if __name__ == "__main__":
    main()
