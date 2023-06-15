from langchain.chains import RetrievalQA
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline, BaseLLM
from constants import CHROMA_SETTINGS, PERSIST_DIRECTORY
import click
import os
from huggingface_hub import hf_hub_download

from constants import CHROMA_SETTINGS

tokenizer_dir = "qwopqwop/KoAlpaca-Polyglot-12.8B-GPTQ"


def load_ko_alpaca(device: str = "cuda") -> BaseLLM:
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

    if device == "cuda":
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
    elif device == "cpu":
        # TODO : Feature/#20
        raise ValueError("We are developing cpu version of koAlpaca model. See Feature/#20")
    else:
        raise ValueError("device type must be cuda or cpu")


def load_openai_model():
    openai_token = os.environ["OPENAI_API_KEY"]
    if openai_token is None:
        raise ValueError("OPENAI_API_KEY is empty.")
    try:
        from langchain.llms import OpenAI
    except ImportError:
        raise ModuleNotFoundError(
            "Could not import OpenAI library. Please install the OpenAI library."
            "pip install openai"
        )
    return OpenAI()


def load_kullm_model(device:str = "cuda"):
    if device == "cuda":
        try:
            from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import transformers library or torch library "
                "Please install the transformers library to "
                "use this embedding model: pip install transformers"
            )
        except Exception:
            raise NameError(f"Could not load model. Check your internet connection.")
        model_name = "nlpai-lab/kullm-polyglot-5.8b-v2"
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=False,
        ).to(device=f"cuda", non_blocking=True)
        pipe = pipeline("text-generation", model=model, tokenizer=model_name, max_new_tokens=100)
        return HuggingFacePipeline(pipeline=pipe)
    elif device == "cpu":
        try:
            from langchain.llms import CTransformers
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import ctransformers library or torch library "
                "Please install the ctransformers library to "
                "pip install ctransformers"
            )
        except Exception:
            raise NameError("Unknown error")

        save_dir = "./models"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        repo_id = "myeolinmalchi/kullm-polyglot-12.8b-v2-GGML"
        bin_filename = "kullm-polyglot-12.8B-v2.ggmlv3.q5_1.bin"
        if not os.path.isfile(save_dir + "/" + bin_filename):
            hf_hub_download(repo_id=repo_id, filename=bin_filename, local_dir=save_dir)
        return CTransformers(model=save_dir + "/" + bin_filename, model_type="gpt_neox")
    else:
        raise ValueError("device type must be cuda or cpu")

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
        llm = load_ko_alpaca(device)
    elif model_type in ["OpenAI", "openai", "Openai"]:
        os.environ["OPENAI_API_KEY"] = openai_token
        llm = load_openai_model()
    elif model_type in ["KULLM", "KuLLM", "kullm"]:
        llm = load_kullm_model(device)
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
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # Interactive questions and answers
    while True:
        query = input("\n질문을 입력하세요: ")
        if query in ["exit", "종료"]:
            break

        # Get the answer from the chain
        res = qa({"query": query})
        answer, docs = res['result'], res['source_documents']

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
