from langchain.prompts import PromptTemplate, ChatPromptTemplate


class RAGchainPromptTemplate(PromptTemplate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # validate template
        if "question" not in self.input_variables:
            raise ValueError("question must be inside prompt as input variable")
        if "passages" not in self.input_variables:
            raise ValueError("passages must be inside prompt as input variable")


class RAGchainChatPromptTemplate(ChatPromptTemplate):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # validate template
        if "question" not in self.input_variables:
            raise ValueError("question must be inside prompt as input variable")
        if "passages" not in self.input_variables:
            raise ValueError("passages must be inside prompt as input variable")
