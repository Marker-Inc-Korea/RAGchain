from KoPrivateGPT.model.base_model_factory import BaseModelFactory
from utils import StoppingCriteriaSub


class KoAlpacaFactory(BaseModelFactory):
    def load_cpu_model(self):
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
        return CTransformers(model="vkehfdl1/KoAlpaca-Polyglot-12.8b-ggml",
                             model_file="KoAlpaca-Polyglot-12.8b-ggml-model-q5_0.bin",
                             model_type="gpt_neox")

    def load_mps_model(self):
        raise NotImplementedError("MPS is not supported yet.")

    def load_cuda_model(self):
        try:
            from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig, AutoModelForCausalLM, \
                LogitsProcessorList, RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor, \
                StoppingCriteria, StoppingCriteriaList
            import torch
            from langchain import HuggingFacePipeline
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import transformers library or torch library "
                "Please install the transformers library to "
                "use this model: pip install transformers"
            )
        except Exception:
            raise NameError(f"Could not load model. Check your internet connection.")

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
