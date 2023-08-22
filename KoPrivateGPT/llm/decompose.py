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

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.llm.base import BaseLLM
from KoPrivateGPT.llm.basic import BasicLLM
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage


class DecomposeLLM(BaseLLM):
    def __init__(self,
                 retrieval: BaseRetrieval,
                 db: BaseDB,
                 model_name: str = "gpt-3.5-turbo",
                 api_base: str = None,
                 decompose_model_name: str = "text-davinci-03",
                 retrieve_size: int = 5,
                 *args, **kwargs):
        self.retrieval = retrieval
        self.db = db
        self.model_name = model_name
        self.decompose_model_name = decompose_model_name
        BasicLLM.set_model(api_base)

    def ask(self, query: str) -> tuple[str, List[Passage]]:
        pass
