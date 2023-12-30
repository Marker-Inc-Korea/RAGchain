from typing import Type, Union, List, Optional, Any

from langchain.prompts.base import StringPromptValue
from langchain.prompts.chat import ChatPromptValueConcrete, ChatPromptValue
from langchain.schema.language_model import LanguageModelInput
from langchain.schema.messages import AnyMessage
from langchain.schema.runnable import RunnableSerializable, RunnableConfig
from langchain.schema.runnable.utils import Input, Output
from pydantic import Field


class LLMLinguaCompressor(RunnableSerializable[LanguageModelInput, str]):
    try:
        from llmlingua import PromptCompressor
    except ImportError:
        raise ImportError("Please install llmlingua first.")

    model_name: str = "NousResearch/Llama-2-7b-hf"
    device_map: str = "cuda"
    model_config: dict = Field(default={})
    open_api_config: dict = Field(default={})
    compressor = PromptCompressor(model_name=model_name,
                                  device_map=device_map,
                                  model_config=model_config,
                                  open_api_config=open_api_config)
    compress_option: dict = Field(default={})

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None,
               **kwargs: Any) -> Output:
        prompt_value = ''
        if isinstance(input, str):
            prompt_value = input
        elif isinstance(input, StringPromptValue):
            prompt_value = input.text
        elif isinstance(input, ChatPromptValueConcrete) or isinstance(input, ChatPromptValue):
            for message in input.messages:
                prompt_value += f'{message.type} : {message.content}\n'
        elif isinstance(input[0], AnyMessage):
            for message in input:
                prompt_value += f'{message.type} : {message.content}\n'
        else:
            raise TypeError(f"Invalid input type: {type(input)}")

        result = self.compressor.compress_prompt(context=[prompt_value],
                                                 **self.compress_option)
        return result['compressed_prompt']

    @property
    def InputType(self) -> Type[Input]:
        """Get the input type for this runnable."""

        return Union[
            str,
            Union[StringPromptValue, ChatPromptValueConcrete],
            List[AnyMessage],
        ]
