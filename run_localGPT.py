from typing import List

import click
from langchain import LLMChain
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

from KoPrivateGPT.pipeline import BasicRunPipeline
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.embed import EmbeddingFactory
from KoPrivateGPT.utils.util import text_modifier
from KoPrivateGPT.utils.vectorDB import Chroma, Pinecone
from config import ChromaOptions, PineconeOptions
from config import MongoDBOptions
from config import Options, PickleDBOptions


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


def hyde_embeddings(llm, base_embedding):
    hyde_prompt = """
    다음 질문에 대하여, 적절한 정보가 주어졌다고 가정하고 대답을 생성하세요.
    질문: {question}
    답변: """
    prompt = PromptTemplate(template=hyde_prompt, input_variables=["question"])
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    return HypotheticalDocumentEmbedder(llm_chain=llm_chain, base_embeddings=base_embedding)


@click.command()
@click.option('--device_type', default='mps', help='device to run on, select gpu, cpu or mps')
@click.option('--retrieval_type', default='bm25', help='retrieval type to use, select vectordb or bm25')
@click.option('--vectordb_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='ko_sroberta_multitask', help='embedding model to use, select OpenAI or KoSimCSE.')
@click.option('--model_name', default='gpt-3.5-turbo', help='model name to use.')
@click.option('--api_base', default=None, help='api base to use.')
@click.option('--db_type', default='mongo_db', help='db type to use, select pickle_db or mongo_db')
def main(device_type, retrieval_type: str, vectordb_type, embedding_type, model_name, api_base, db_type: str):
    if vectordb_type == text_modifier('chroma'):
        vectordb = Chroma(ChromaOptions.persist_dir, ChromaOptions.collection_name)
    elif vectordb_type == text_modifier('pinecone'):
        vectordb = Pinecone(PineconeOptions.index_name,
                            PineconeOptions.namespace,
                            PineconeOptions.dimension)
    else:
        raise ValueError("vectordb type is not valid")

    pipeline = BasicRunPipeline(
        retrieval_type=(retrieval_type, {"save_path": Options.bm25_db_dir,
                                         "vectordb": vectordb,
                                         "embedding_type": EmbeddingFactory(embed_type=embedding_type,
                                                                            device_type=device_type).get(),
                                         "device_type": device_type}),
        llm_type=("basic_llm", {"model_name": model_name, "api_base": api_base}),
        db_type=(db_type, {
            'save_path': PickleDBOptions.save_path,
            "mongo_url": MongoDBOptions.mongo_url,
            "db_name": MongoDBOptions.db_name,
            "collection_name": MongoDBOptions.collection_name
        })
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
