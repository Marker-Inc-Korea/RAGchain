from typing import List

import click
from langchain import LLMChain
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate

from KoPrivateGPT.options import Options, PickleDBOptions
from KoPrivateGPT.options.config import MongoDBOptions
from KoPrivateGPT.pipeline import BasicRunPipeline
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.embed import EmbeddingFactory


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
def main(device_type, retrieval_type: str, vectordb_type, embedding_type, model_name, api_base):
    pipeline = BasicRunPipeline(
        retrieval_type=(retrieval_type, {"save_path": Options.bm25_db_dir,
                                         "vectordb_type": vectordb_type,
                                         "embedding_type": EmbeddingFactory(embed_type=embedding_type,
                                                                            device_type=device_type).get(),
                                         "device_type": device_type}),
        llm_type=("basic_llm", {"model_name": model_name, "api_base": api_base})
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
