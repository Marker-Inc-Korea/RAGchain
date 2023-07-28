from langchain import LLMChain
from langchain.chains import RetrievalQA, HypotheticalDocumentEmbedder
from langchain.prompts import PromptTemplate
import click

from utils import slice_stop_words
from vectorDB import DB
from dotenv import load_dotenv
from embed import Embedding
from model import load_model


def make_qa(llm, retriever):
    prompt_template = """주어진 정보를 바탕으로 질문에 답하세요. 답을 모른다면 답을 지어내려고 하지 말고 모른다고 답하세요. 
            질문 이외의 상관 없는 답변을 하지 마세요. 반드시 한국어로 답변하세요.

            {context}

            질문: {question}
            한국어 답변:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
                                     chain_type_kwargs=chain_type_kwargs)
    return qa


def get_answer(retrieval_qa, query):
    # Get the answer from the chain
    res = retrieval_qa({"query": query})
    answer, docs = res['result'], res['source_documents']

    return query, answer, docs


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
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE', help='embedding model to use, select OpenAI or KoSimCSE.')
def main(device_type, model_type, db_type, embedding_type):
    load_dotenv()

    llm = load_model(model_type, device_type=device_type)

    embeddings = Embedding(embed_type=embedding_type, device_type=device_type).embedding()
    embeddings = hyde_embeddings(llm, embeddings)
    # load the vectorstore
    db = DB(db_type, embeddings).load()
    retriever = db.as_retriever(search_kwargs={"k": 4})
    retrieval_qa = make_qa(llm, retriever)
    while True:
        query = input("질문을 입력하세요: ")
        if query in ["exit", "종료"]:
            break
        query, answer, docs = get_answer(retrieval_qa, query)
        answer = slice_stop_words(answer, ["Question :", "question:"])
        print_query_answer(query, answer)
        print_docs(docs)


if __name__ == "__main__":
    main()
