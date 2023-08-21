import os

import gradio as gr
from dotenv import load_dotenv

from KoPrivateGPT.options import Options, ChromaOptions, PineconeOptions, PickleDBOptions, MongoDBOptions
from KoPrivateGPT.pipeline import BasicIngestPipeline, BasicRunPipeline
from KoPrivateGPT.utils.embed import EmbeddingFactory
from KoPrivateGPT.utils.util import slice_stop_words

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

    bm25 = ("bm25", {"save_path": Options.bm25_db_dir})
    pickle = ("pickle_db", {"save_path": PickleDBOptions.save_path})
    mongo = ("mongo_db", {"mongo_url": MongoDBOptions.mongo_url,
                          "db_name": MongoDBOptions.db_name,
                          "collection_name": MongoDBOptions.collection_name})
    chroma = ("vector_db", {"vectordb_type": "chroma",
                            "embedding": EmbeddingFactory(embed_type=embed, device_type=device).get()})
    pinecone = ("vector_db", {"vectordb_type": "pinecone",
                              "embedding": EmbeddingFactory(embed_type=embed, device_type=device).get()})

    if retrieval == "bm25":
        answer_pipeline = BasicRunPipeline(retrieval_type=bm25,
                                           llm_type=("basic_llm", {"model_name": MODEL_NAME, "api_base": None}))
        pre_retrieval = bm25
    elif retrieval == "vector_db-chroma":
        answer_pipeline = BasicRunPipeline(retrieval_type=chroma,
                                           llm_type=("basic_llm", {"model_name": MODEL_NAME, "api_base": None}))
        pre_retrieval = chroma
    else:
        answer_pipeline = BasicRunPipeline(retrieval_type=pinecone)

    if db == "pickle_db":
        ingest_pipeline = BasicIngestPipeline(file_loader_type=("file_loader", {}), db_type=pickle,
                                              retrieval_type=pre_retrieval)
    else:
        ingest_pipeline = BasicIngestPipeline(file_loader_type=("file_loader", {}), db_type=mongo,
                                              retrieval_type=pre_retrieval)

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
                choices=["pickle", "mongodb"],
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
