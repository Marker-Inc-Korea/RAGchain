"""
This code is inspired by Visconde paper and its github repo.
@inproceedings{10.1007/978-3-031-28238-6_44,
author = {Pereira, Jayr and Fidalgo, Robson and Lotufo, Roberto and Nogueira, Rodrigo},
title = {Visconde: Multi-Document QA With&nbsp;GPT-3 And&nbsp;Neural Reranking},
year = {2023},
isbn = {978-3-031-28237-9},
publisher = {Springer-Verlag},
address = {Berlin, Heidelberg},
url = {https://doi.org/10.1007/978-3-031-28238-6_44},
doi = {10.1007/978-3-031-28238-6_44},
booktitle = {Advances in Information Retrieval: 45th European Conference on Information Retrieval, ECIR 2023, Dublin, Ireland, April 2–6, 2023, Proceedings, Part II},
pages = {534–543},
numpages = {10},
location = {Dublin, Ireland}
}
"""
from typing import List
import openai

from KoPrivateGPT.llm.basic import BasicLLM


class QueryDecomposition:
    decompose_prompt = """Decompose a question in self-contained sub-questions. Use \"The question needs no decomposition\" when no decomposition is needed.
    
    Example 1:
    
    Question: Is Hamlet more common on IMDB than Comedy of Errors?
    Decompositions: 
    1: How many listings of Hamlet are there on IMDB?
    2: How many listing of Comedy of Errors is there on IMDB?
    
    Example 2:
    
    Question: Are birds important to badminton?
    
    Decompositions:
    The question needs no decomposition
    
    Example 3:
    
    Question: Is it legal for a licensed child driving Mercedes-Benz to be employed in US?
    
    Decompositions:
    1: What is the minimum driving age in the US?
    2: What is the minimum age for someone to be employed in the US?
    
    Example 4:
    
    Question: Are all cucumbers the same texture?
    
    Decompositions:
    The question needs no decomposition
    
    Example 5:
    
    Question: Hydrogen's atomic number squared exceeds number of Spice Girls?
    
    Decompositions:
    1: What is the atomic number of hydrogen?
    2: How many Spice Girls are there?
    
    Example 6:
    
    Question: {0}
    
    Decompositions:"
    """

    def __init__(self, model_name: str = "text-davinci-003", api_base: str = None):
        self.model_name = model_name
        BasicLLM.set_model(api_base)

    def decompose(self, query: str) -> List[str]:
        """
        decompose query to little piece of questions.
        :param query: str, query to decompose.
        :return: List[str], list of decomposed query. Return empty list if query is not decomposable.
        """
        res = self.generate(self.decompose_prompt.format(query))
        if res.lower().strip() == "the question needs no decomposition.":
            return [query]
        try:
            questions = [l for l in res.splitlines() if l != ""]
            questions = [q.split(':')[1].strip() for q in questions]
            return questions
        except:
            return [query]

    def generate(self, prompt, max_tokens=250, temperature=0):
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        return response['choices'][0]['text']
