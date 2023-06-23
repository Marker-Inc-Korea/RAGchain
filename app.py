from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY

import gradio as gr

from ingest import load_single_document, embedding_open_api
from run_localGPT import load_ko_alpaca, load_openai_model, load_kullm_model

from dotenv import load_dotenv
import os

from utils import slice_stop_words

load_dotenv()

STOP_WORDS = ["#", "답변:", "응답:", "\n", "맥락:", "?"]

PROMPT_TEMPLATE = """주어진 정보를 바탕으로 질문에 답하세요. 답을 모른다면 답을 지어내려고 하지 말고 모른다고 답하세요. 
    질문 이외의 상관 없는 답변을 하지 마세요. 반드시 한국어로 답변하세요.

    {context}

    질문: {question}
    한국어 답변:"""

device = "cuda"
model_type = "OpenAI"
llm = load_openai_model()

embeddings = embedding_open_api()
#HuggingFaceInstructEmbeddings(model_name="BM-K/KoSimCSE-roberta-multitask",
#                                           model_kwargs={"device": device})


def ingest(files) -> str:
    file_paths = [f.name for f in files]
    documents = [load_single_document(path) for path in file_paths]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    texts = text_splitter.split_documents(documents)
    db = Chroma.from_documents(texts, embeddings, persist_directory=PERSIST_DIRECTORY, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None
    return "Ingest Done"


def get_answer(text):
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings,
                client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
                                     chain_type_kwargs=chain_type_kwargs)

    # Get the answer from the chain
    res = qa({"query": text})
    answer, docs = res['result'], res['source_documents']
    answer = slice_stop_words(answer, STOP_WORDS)
    # Print the result
    print("\n\n> 질문:")
    print(text)
    print("\n> 대답:")
    print(answer)

    # # Print the relevant sources used for the answer
    print("----------------------------------참조한 문서---------------------------")
    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
    print("----------------------------------참조한 문서---------------------------")
    document_sources = ",\n".join([doc.metadata["source"].split("/")[-1] for doc in docs])
    return answer, document_sources


with gr.Blocks(analytics_enabled=False) as demo:
    gr.HTML(
        f"""<div style="text-align: center; max-width: 500px; margin: 0 auto;">
                <div>
                <h1>KoPrivateGPT with {model_type}</h1>
                </div>
                <div>
                Demo runs on {device}
                </div>
            </div>"""
    )
    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(label="질문 내용", placeholder="질문을 입력하세요", interactive=True,lines=17, max_lines=17)
            question_btn = gr.Button("질문하기")

        with gr.Column(scale=3):
            answer_result = gr.Textbox(label="답변 내용", placeholder="답변을 출력합니다.", interactive=False,lines=20, max_lines=20)

    gr.HTML(
        """<h2 style="text-align: center;"><br>파일 업로드하기<br></h2>"""
    )
    upload_files = gr.Files()
    ingest_status = gr.Textbox(value="", label="Ingest Status")
    ingest_button = gr.Button("Ingest")
    ingest_button.click(ingest, inputs=[upload_files], outputs=[ingest_status])

    document_sources = gr.Textbox(label="참조한 문서", placeholder="참조한 문서를 출력합니다.", interactive=False)
    question_btn.click(get_answer, inputs=[query], outputs=[answer_result, document_sources])

demo.launch(share=False, debug=True, server_name="0.0.0.0")
