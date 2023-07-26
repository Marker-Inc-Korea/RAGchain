from langchain import LLMChain
from typing import Tuple, List
from langchain.prompts import PromptTemplate
import click
from langchain.schema import Document

from options import Options
from retrieve import LangchainRetriever, BM25Retriever
from retrieve.base import BaseRetriever
from utils import slice_stop_words
from dotenv import load_dotenv
from embed import Embedding
from model import load_model


def make_llm_chain(llm):
    prompt_template = """주어진 정보를 바탕으로 질문에 답하세요. 답을 모른다면 답을 지어내려고 하지 말고 모른다고 답하세요. 
                질문 이외의 상관 없는 답변을 하지 마세요. 반드시 한국어로 답변하세요.

                {context}

                질문: {question}
                한국어 답변:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain


def get_answer(chain: LLMChain, retriever: BaseRetriever, query: str) -> Tuple[str, List[Document]]:
    # Get the answer from the chain
    retrieved_documents = retriever.retrieve(query, top_k=4)
    document_texts = "\n\n".join([document.page_content for document in retrieved_documents])
    answer = chain.run(context=document_texts, question=query)
    return answer, retrieved_documents


def print_query_answer(query, answer):
    # Print the result
    print("\n\n> 질문:")
    print(query)
    print("\n> 대답:")
    print(answer)


def print_docs(docs):
    # Print the relevant sources used for the answer
    print("----------------------------------참조한 문서---------------------------")
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
    print("----------------------------------참조한 문서---------------------------")


@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--model_type', default='koAlpaca', help='model to run on, select koAlpaca or openai')
@click.option('--retriever_type', default='langchain', help='retriever type to use, select langchain or bm25')
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE', help='embedding model to use, select OpenAI or KoSimCSE.')
def main(device_type, model_type, retriever_type, db_type, embedding_type):
    load_dotenv()

    llm = load_model(model_type, device_type=device_type)
    chain = make_llm_chain(llm)
    # load the vectorstore
    if retriever_type in ['bm25', 'BM25']:
        retriever = BM25Retriever.load(Options.bm25_db_dir)
    else:
        embeddings = Embedding(embed_type=embedding_type, device_type=device_type).embedding()
        retriever = LangchainRetriever.load(db_type=db_type, embedding=embeddings)

    while True:
        query = input("질문을 입력하세요: ")
        if query in ["exit", "종료"]:
            break
        answer, docs = get_answer(chain, retriever, query)
        answer = slice_stop_words(answer, ["Question :", "question:"])
        print_query_answer(query, answer)
        print_docs(docs)


if __name__ == "__main__":
    main()
