import os
from typing import List

import click
from dotenv import load_dotenv

from KoPrivateGPT.pipeline import BasicRunPipeline
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.embed import EmbeddingFactory
from KoPrivateGPT.utils.util import text_modifier
from KoPrivateGPT.utils.vectorDB import Chroma, Pinecone
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

def select_vectordb(vectordb_type: str):
    load_dotenv()
    if vectordb_type == text_modifier('chroma'):
        vectordb = Chroma(ChromaOptions.persist_dir, ChromaOptions.collection_name)
    elif vectordb_type == text_modifier('pinecone'):
        vectordb = Pinecone(os.getenv('PINECONE_API_KEY'),
                            os.getenv('PINECONE_ENV'),
                            PineconeOptions.index_name,
                            PineconeOptions.namespace,
                            PineconeOptions.dimension)
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
    vectordb = select_vectordb(vectordb_type)
    pipeline = BasicRunPipeline(
        retrieval_type=(retrieval_type, {"save_path": Options.bm25_db_dir,
                                         "vectordb": vectordb,
                                         "embedding": EmbeddingFactory(embed_type=embedding_type,
                                                                       device_type=device_type).get()
                                         }),
        llm_type=("basic_llm", {"model_name": model_name, "api_base": api_base}),
    )
    while True:
        query = input("질문을 입력하세요: ")
        if query in ["exit", "종료"]:
            break
        answer, passages = pipeline.run(query)
        print_query_answer(query, answer)
        print_docs(passages)


if __name__ == "__main__":
    main()
