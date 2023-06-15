from langchain.chains import RetrievalQA
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
import click
import os

from constants import CHROMA_SETTINGS

tokenizer_dir = "qwopqwop/KoAlpaca-Polyglot-12.8B-GPTQ"


def load_model(model_type: str = "koAlpaca"):
    try:
        from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM, \
            LogitsProcessorList, RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor
        import torch
    except ImportError:
        raise ModuleNotFoundError(
            "Could not import transformers library or torch library "
            "Please install the transformers library to "
            "use this embedding model: pip install transformers"
        )
    except Exception:
        raise NameError(f"Could not load model. Check your internet connection.")

    if model_type == "koAlpaca":
        model_id = "beomi/polyglot-ko-12.8b-safetensors"  # safetensors 컨버팅된 레포
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
        # define repetition penalty (not fixed)
        repetition_penalty = float(1.2)
        logits_processor = LogitsProcessorList(
            [RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty), NoRepeatNGramLogitsProcessor(5)])
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100,
                        logits_processor=logits_processor)
        return HuggingFacePipeline(pipeline=pipe)
    elif model_type == "openai":
        try:
            from langchain.llms import OpenAI
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import OpenAI library. Please install the OpenAI library."
            )
        return OpenAI()
    else:
        raise ValueError(f"Invalid model type: {model_type}")

# @click.command()
# @click.option('--device_type', default='gpu', help='device to run on, select gpu or cpu')
# def main(device_type, ):
#     # load the instructorEmbeddings
#     if device_type in ['cpu', 'CPU']:
#         device='cpu'
#     else:
#         device='cuda'
 
    
 ## for M1/M2 users:

@click.command()
@click.option('--device_type', default='cuda', help='device to run on, select gpu, cpu or mps')
@click.option('--model_type', default='koAlpaca', help='model to run on, select koAlpaca or openai')
@click.option('--openai-token', help='openai token')
def main(device_type, model_type, openai_token):
    # load the instructorEmbeddings
    if device_type in ['cpu', 'CPU']:
        device='cpu'
    elif device_type in ['mps', 'MPS']:
        device='mps'
    else:
        device='cuda'

    if model_type in ['koAlpaca', 'KoAlpaca', 'koalpaca', 'Ko-alpaca']:
        model_type = 'koAlpaca'
    elif model_type in ["OpenAI", "openai", "Openai"]:
        model_type = "openai"
        os.environ["OPENAI_API_KEY"] = openai_token
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    print(f"Running on: {device}")

    embeddings = HuggingFaceInstructEmbeddings(model_name = "BM-K/KoSimCSE-roberta-multitask", model_kwargs={"device": device})
    # load the vectorstore
    db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    # Prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]
    # load the LLM for generating Natural Language responses. 
    llm = load_model(model_type)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\n질문을 입력하세요: ")
        if query in ["exit", "종료"]:
            break

        # Get the answer from the chain
        res = qa({"query": query})
        answer, docs = res['result'], res['source_documents']

        if "question:" in answer:
            answer = answer.split("question:")[:1]
        elif "Question:" in answer:
            answer = answer.split("Question:")[:1]

        # Print the result
        print("\n\n> 질문:")
        print(query)
        print("\n> 대답:")
        print(answer)

        # # Print the relevant sources used for the answer
        print("----------------------------------참조한 문서---------------------------")
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
        print("----------------------------------참조한 문서---------------------------")


if __name__ == "__main__":
    main()
