from KoPrivateGPT.model.koalpaca import KoAlpacaFactory
from KoPrivateGPT.model.kullm import KuLLMFactory
from KoPrivateGPT.model.openai import OpenaiFactory


def load_model(model_type: str, device_type: str = 'cuda'):
    if model_type in ['koAlpaca', 'KoAlpaca', 'koalpaca', 'Ko-alpaca']:
        factory = KoAlpacaFactory()
    elif model_type in ["OpenAI", "openai", "Openai"]:
        factory = OpenaiFactory()
    elif model_type in ["KULLM", "KuLLM", "kullm"]:
        factory = KuLLMFactory()
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    llm = factory.load(device_type)
    return llm
