from langchain.text_splitter import RecursiveCharacterTextSplitter

import gradio as gr

from ingest import load_single_document, split_documents, ingest_texts
from model import load_model
from dotenv import load_dotenv

from run_localGPT import make_qa, get_answer, hyde_embeddings
from utils import slice_stop_words
from vectorDB import DB
from embed import Embedding

load_dotenv()

STOP_WORDS = ["#", "답변:", "응답:", "맥락:", "?"]

device = "cuda"
model_type = "OpenAI"
llm = load_model(model_type)
embedding_type = "OpenAI"
embeddings = Embedding(embed_type=embedding_type).embedding()
embeddings = hyde_embeddings(llm, embeddings)
db = DB('pinecone', embeddings).load()
retriever = db.as_retriever()


def ingest(files) -> str:
    file_paths = [f.name for f in files]
    documents = [load_single_document(path) for path in file_paths]
    texts = split_documents(documents)
    ingest_texts('cuda', 'pinecone', 'OpenAI', texts)
    return "Ingest Done"


def make_answer(text):
    qa = make_qa(llm, retriever)
    # Get the answer from the chain
    q, answer, docs = get_answer(qa, text)
    answer = slice_stop_words(answer, STOP_WORDS)
    document_source = ",\n".join([doc.metadata["source"].split("/")[-1] for doc in docs])
    return answer, document_source


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
    question_btn.click(make_answer, inputs=[query], outputs=[answer_result, document_sources])

demo.launch(share=False, debug=True, server_name="0.0.0.0")
