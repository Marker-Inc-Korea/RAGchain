from model.base_model_factory import BaseModelFactory


class KuLLMFactory(BaseModelFactory):
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

        return CTransformers(model="vkehfdl1/KuLLM-Polyglot-12.8b-v2-ggml",
                             model_file="KuLLM-Polyglot-12.8b-v2-ggml-q5_0.bin",
                             model_type="gpt_neox")

    def load_mps_model(self):
        raise NotImplementedError("KuLLM does not support MPS")

    def load_cuda_model(self):
        try:
            from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
            from langchain import HuggingFacePipeline
            import torch
        except ImportError:
            raise ModuleNotFoundError(
                "Could not import transformers library or torch library "
                "Please install the transformers library to "
                "use this model: pip install transformers"
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
