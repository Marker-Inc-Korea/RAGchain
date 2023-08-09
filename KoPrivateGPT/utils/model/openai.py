from KoPrivateGPT.utils.model.base_model_factory import BaseModelFactory
import os


class OpenaiFactory(BaseModelFactory):
    def load_cpu_model(self):
        print("Using OpenAI API. This is not local model.")
        return self.__load_model_openai()

    def load_mps_model(self):
        print("Using OpenAI API. This is not local model.")
        return self.__load_model_openai()

    def load_cuda_model(self):
        print("Using OpenAI API. This is not local model.")
        return self.__load_model_openai()

    def __load_model_openai(self):
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
        return OpenAI(max_tokens=1024, model_name='gpt-3.5-turbo')
