import os

import gradio as gr
from dotenv import load_dotenv

from KoPrivateGPT.options import Options
from KoPrivateGPT.pipeline import BasicIngestPipeline, BasicRunPipeline
from KoPrivateGPT.utils.embed import EmbeddingFactory
from KoPrivateGPT.utils.util import slice_stop_words

load_dotenv()

STOP_WORDS = ["#", "답변:", "응답:", "맥락:", "?"]

DEVICE = "mps"
MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_TYPE = "openai"
DB_TYPE = "chroma"
embeddings = EmbeddingFactory(embed_type=EMBEDDING_TYPE, device_type=DEVICE).get()

GRADIO_DB_PATH = os.path.join(Options.root_dir, )
answer_pipeline = BasicRunPipeline(retrieval_type=("bm25", {"save_path": Options.bm25_db_dir}),
                                   llm_type=("basic_llm", {"model_name": MODEL_NAME, "api_base": None}))
ingest_pipeline = BasicIngestPipeline(file_loader_type=("file_loader", {}),
                                      retrieval_type=("bm25", {"save_path": Options.bm25_db_dir}))


def ingest(files) -> str:
    # TODO : add file cache for gradio version : Feature/#94
    dir_name = os.path.dirname(files[0].name)
    ingest_pipeline.run(target_dir=dir_name)
    return "Ingest Done"


def make_answer(text):
    answer, passages = answer_pipeline.run(text)
    answer = slice_stop_words(answer, STOP_WORDS)
    document_source = ",\n".join([doc.filepath.split("/")[-1] for doc in passages])
    content = "\n-------------------------------------------------\n".join([doc.content for doc in passages])
    return answer, document_source, content


with gr.Blocks(analytics_enabled=False) as demo:
    gr.HTML(
        f"""<div style="text-align: center; max-width: 500px; margin: 0 auto;">
                <div>
                <h1>KoPrivateGPT with {MODEL_NAME}</h1>
                </div>
                <div>
                Demo runs on {DEVICE}
                </div>
            </div>"""
    )
    with gr.Row():
        with gr.Column(scale=2):
            query = gr.Textbox(label="질문 내용", placeholder="질문을 입력하세요", interactive=True, lines=17, max_lines=17)
            question_btn = gr.Button("질문하기")

        with gr.Column(scale=3):
            answer_result = gr.Textbox(label="답변 내용", placeholder="답변을 출력합니다.", interactive=False, lines=20,
                                       max_lines=20)

    gr.HTML(
        """<h2 style="text-align: center;"><br>파일 업로드하기<br></h2>"""
    )
    upload_files = gr.Files()
    ingest_status = gr.Textbox(value="", label="Ingest Status")
    ingest_button = gr.Button("Ingest")
    ingest_button.click(ingest, inputs=[upload_files], outputs=[ingest_status])

    document_sources = gr.Textbox(label="참조한 문서", placeholder="참조한 문서를 출력합니다.", interactive=False)
    contents = gr.Textbox(label="참조한 문서 내용", placeholder="참조한 문서 내용을 출력합니다.", interactive=False)
    question_btn.click(make_answer, inputs=[query], outputs=[answer_result, document_sources, contents])


demo.launch(share=False, debug=True, server_name="0.0.0.0")
