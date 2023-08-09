from typing import List

from langchain import LLMChain
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
import click

from KoPrivateGPT.DB.pickle_db import PickleDB
from KoPrivateGPT.llm.basic import BasicLLM
from KoPrivateGPT.options import Options, DBOptions
from KoPrivateGPT.retrieval import VectorDBRetrieval, BM25Retrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.util import slice_stop_words
from dotenv import load_dotenv
from KoPrivateGPT.utils.embed import Embedding
from KoPrivateGPT.utils.model import load_model


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
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--model_type', default='koAlpaca', help='model to run on, select koAlpaca or openai')
@click.option('--retrieval_type', default='vectordb', help='retrieval type to use, select vectordb or bm25')
@click.option('--vectordb_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE', help='embedding model to use, select OpenAI or KoSimCSE.')
def main(device_type, model_type, retrieval_type, vectordb_type, embedding_type):
    load_dotenv()

    # Load DB
    db = PickleDB(DBOptions.save_path)
    db.load()

    # load the retrieval
    if retrieval_type in ['bm25', 'BM25']:
        retrieval = BM25Retrieval(Options.bm25_db_dir, db=db)
    else:
        embeddings = Embedding(embed_type=embedding_type, device_type=device_type)
        retrieval = VectorDBRetrieval(vectordb_type=vectordb_type, embedding=embeddings, db=db)

    llm = BasicLLM(retrieval=retrieval, model_type=model_type, device_type=device_type)

    while True:
        query = input("질문을 입력하세요: ")
        if query in ["exit", "종료"]:
            break

        answer, passages = llm.ask(query)
        answer = slice_stop_words(answer, ["Question :", "question:"])
        print_query_answer(query, answer)
        print_docs(passages)


if __name__ == "__main__":
    main()
