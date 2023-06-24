from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline, BaseLLM
from langchain.prompts import PromptTemplate
import click
import os

from db import DB
from utils import StoppingCriteriaSub
from dotenv import load_dotenv
from embedding import EMBEDDING

def load_ko_alpaca(device: str = "cuda") -> BaseLLM:
    try:
        from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM, \
            LogitsProcessorList, RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor, \
            StoppingCriteria, StoppingCriteriaList
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
        stop_words = ["!"]
        stop_words_ids = [tokenizer(stop_word, return_tensors='pt')['input_ids'].squeeze() for stop_word in stop_words]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids, encounters=1)])
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100,
                        logits_processor=logits_processor, stopping_criteria=stopping_criteria,
                        eos_token_id=2, pad_token_id=2)
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
        return CTransformers(model="vkehfdl1/KoAlpaca-Polyglot-12.8b-ggml", model_file="KoAlpaca-Polyglot-12.8b-ggml-model-q5_0.bin",
                             model_type="gpt_neox")

    else:
        raise ValueError("device type must be cuda or cpu")


def load_openai_model() -> BaseLLM:
    openai_token = os.environ["OPENAI_API_KEY"]
    if openai_token is None:
        raise ValueError("OPENAI_API_KEY is empty. Set OPENAI_API_KEY at .env file")
    try:
        from langchain.llms import OpenAI
    except ImportError:
        raise ModuleNotFoundError(
            "Could not import OpenAI library. Please install the OpenAI library."
            "pip install openai"
        )
    return OpenAI(max_tokens=1024)


def load_kullm_model(device: str = "cuda") -> BaseLLM:
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

        return CTransformers(model="vkehfdl1/KuLLM-Polyglot-12.8b-v2-ggml", model_file="KuLLM-Polyglot-12.8b-v2-ggml-q5_0.bin",
                             model_type="gpt_neox")
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
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='KoSimCSE', help='embedding model to use, select OpenAI or KoSimCSE')
def main(device_type, model_type, db_type, embedding_type):
    load_dotenv()
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
        llm = load_openai_model()
    elif model_type in ["KULLM", "KuLLM", "kullm"]:
        llm = load_kullm_model(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    print(f"Running on: {device}")

    embeddings = EMBEDDING(embedding_type).embedding()

    # load the vectorstore
    db = DB(db_type, embeddings).load()
    retriever = db.as_retriever()
    # Prepare the LLM
    # callbacks = [StreamingStdOutCallbackHandler()]
    # load the LLM for generating Natural Language responses.
    prompt_template = """주어진 정보를 바탕으로 질문에 답하세요. 답을 모른다면 답을 지어내려고 하지 말고 모른다고 답하세요. 
    질문 이외의 상관 없는 답변을 하지 마세요. 반드시 한국어로 답변하세요.
    
    {context}
    
    질문: {question}
    한국어 답변:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True,
                                     chain_type_kwargs=chain_type_kwargs)
    # Interactive questions and answers
    while True:
        query = input("\n질문을 입력하세요: ")
        if query in ["exit", "종료"]:
            break

        # Get the answer from the chain
        res = qa({"query": query})
        answer, docs = res['result'], res['source_documents']

        stop_words = ["question:", "Questions:"]
        for stop_word in stop_words:
            if stop_word in answer:
                answer = answer.split(stop_word)[:1]

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
