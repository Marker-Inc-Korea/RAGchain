import os

import gradio as gr
from dotenv import load_dotenv

from RAGchain.DB import MongoDB, PickleDB
from RAGchain.llm.basic import BasicLLM
from RAGchain.pipeline import BasicIngestPipeline, BasicRunPipeline
from RAGchain.preprocess.loader import FileLoader
from RAGchain.retrieval import BM25Retrieval
from RAGchain.utils.util import slice_stop_words
from config import Options, PickleDBOptions, MongoDBOptions
from run_localGPT import select_vectordb

load_dotenv()

STOP_WORDS = ["#", "답변:", "응답:", "맥락:", "?"]
GRADIO_DB_PATH = os.path.join(Options.root_dir, )
model_dict = {"basic_llm": "gpt-3.5-turbo", "rerank_llm": "gpt-3.5-turbo"}


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


def setting(device, embed, db, retrieval):
    global answer_pipeline, ingest_pipeline

    MODEL_NAME = model_dict["basic_llm"]  # TODO : add rerank_llm model

    bm25 = BM25Retrieval(Options.bm25_db_dir)
    pickle = PickleDB(PickleDBOptions.save_path)
    mongo = MongoDB(MongoDBOptions.mongo_url, MongoDBOptions.db_name, MongoDBOptions.collection_name)

    if db == "pickle_db":
        pre_db = pickle
    elif db == "mongo_db":
        pre_db = mongo
    else:
        raise ValueError("db type is not valid")

    if retrieval == "bm25":
        answer_pipeline = BasicRunPipeline(retrieval=bm25,
                                           llm=BasicLLM(bm25, model_name=MODEL_NAME, api_base=None))
        pre_retrieval = bm25
    elif retrieval == "vector_db-chroma":
        vectordb_instance = select_vectordb('chroma', embedding_type=embed, device_type=device)
        answer_pipeline = BasicRunPipeline(retrieval=vectordb_instance,
                                           llm=BasicLLM(bm25, model_name=MODEL_NAME, api_base=None))
        pre_retrieval = vectordb_instance
    elif retrieval == "vector_db-pinecone":
        vectordb_instance = select_vectordb('pinecone', embedding_type=embed, device_type=device)
        answer_pipeline = BasicRunPipeline(retrieval=vectordb_instance)
        pre_retrieval = vectordb_instance
    else:
        raise ValueError("retrieval type is not valid")

    ingest_pipeline = BasicIngestPipeline(file_loader=FileLoader("", hwp_host_url=Options.HwpConvertHost),
                                          db=pre_db,
                                          retrieval=pre_retrieval)

    return 'setting done'


with gr.Blocks(analytics_enabled=False) as demo:
    gr.HTML(
        f"""<div style="text-align: center; max-width: 500px; margin: 0 auto;">
                <div>
                <h1>KoPrivateGPT with gpt-3.5-turbo</h1>
                </div>
                <div>
                Demo runs on selected device type
                </div>
            </div>"""
    )
    with gr.Row():
        with gr.Column(scale=2):
            device_type = gr.Dropdown(
                label="Device Type",
                choices=["cpu", "mps", "cuda"],
                info="디바이스 타입을 설정하세요",
                interactive=True
            )
        with gr.Column(scale=2):
            embedding_type = gr.Dropdown(
                label="Embedding Type",
                choices=["openai", "kosimcse", "ko_sroberta_multitask"],
                info="임베딩 타입을 설정하세요",
                interactive=True
            )
        with gr.Column(scale=2):
            db_type = gr.Dropdown(
                label="DB Type",
                choices=["pickle_db", "mongo_db"],
                info="DB 타입을 설정하세요",
                interactive=True
            )
        with gr.Column(scale=2):
            retrieval_type = gr.Dropdown(
                label="retrieval Type",
                choices=["bm25", "vector_db-chroma", "vector_db-pinecone"],
                info="retrieval 타입을 설정하세요",
                interactive=True
            )
        """
        with gr.Column(scale=2):
            model_name = gr.Dropdown(
                label="Model Name",
                choices=["basic_llm", "rerank_llm"],
                info="모델을 설정하세요",
                interactive=True
            )
        """
        with gr.Column(scale=2):
            setting_btn = gr.Button("설정하기")
            done = gr.Textbox(label="설정 상태", placeholder="설정 상태를 표기합니다.", interactive=False, lines=1,
                              max_lines=1)

    setting_btn.click(setting, inputs=[device_type, embedding_type, db_type, retrieval_type],
                      outputs=[done])

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
