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
from collections import OrderedDict
from copy import deepcopy
from typing import List, Callable

from KoPrivateGPT.llm.base import BaseLLM
from KoPrivateGPT.reranker import MonoT5Reranker
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils.query_decompose import QueryDecomposition
from KoPrivateGPT.utils.util import set_api_base


class ViscondeLLM(BaseLLM):
    strategyqa_prompt = """For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Just answer yes or no.
    
    Example 1:
    
    [Document 1]: 
    Title: San Tropez (song). 
    Content: \"San Tropez\" is the fourth track from the album Meddle by the band Pink Floyd. 
    This song was one of several to be considered for the band's \"best of\" album, Echoes: The Best of Pink Floyd.
    
    [Document 2]: 
    Title: French Riviera. 
    Content: The French Riviera (known in French as the Côte d'Azur [kot daˈzyʁ]; Occitan: Còsta d'Azur [
    ˈkɔstɔ daˈzyɾ]; literal translation \"Azure Coast\") is the Mediterranean coastline of the southeast corner of 
    France. There is no official boundary, but it is usually considered to extend from Cassis, Toulon or Saint-Tropez 
    on the west to Menton at the France–Italy border in the east, where the Italian Riviera joins. The coast is 
    entirely within the Provence-Alpes-Côte d'Azur (Région Sud) region of France. The Principality of Monaco is a 
    semi-enclave within the region, surrounded on three sides by France and fronting the Mediterranean.
    
    [Document 3]: 
    Title: Moon Jae-in. 
    Content: Moon also promised transparency in his presidency, moving the presidential residence from the palatial and 
    isolated Blue House to an existing government complex in downtown Seoul.
    
    [Document 4]: 
    Title: Saint-Tropez. 
    Content: Saint-Tropez (US: /ˌsæn troʊˈpeɪ/ SAN-troh-PAY, French: [sɛ̃ tʁɔpe]; Occitan: Sant-Tropetz , pronounced [san(t) tʀuˈpes]) is a town on the French Riviera, 
    68 kilometres (42 miles) west of Nice and 100 kilometres (62 miles) east of Marseille in the Var department of 
    the Provence-Alpes-Côte d'Azur region of Occitania, Southern France.
    

    Question: Did Pink Floyd have a song about the French Riviera?
    Explanation: According to [Document 1], \"San Tropez\" is a song by Pink Floyd about 
    the French Riviera. This is further supported by [Document 4], which states that Saint-Tropez is a town on the French Riviera. 
    Therefore, the answer is yes
    Answer: yes.
    
    """

    def __init__(self,
                 retrieval: BaseRetrieval,
                 model_name: str = "text-davinci-003",
                 api_base: str = None,
                 decompose_model_name: str = "text-davinci-003",
                 retrieve_size: int = 50,
                 use_passage_count: int = 3,
                 prompt: str = None,
                 stream_func: Callable[[str], None] = None,
                 *args, **kwargs):
        super().__init__(retrieval)
        self.model_name = model_name
        self.decompose_model_name = decompose_model_name
        set_api_base(api_base)
        self.api_base = api_base
        self.retrieve_size = retrieve_size
        self.use_passage_count = use_passage_count
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = self.strategyqa_prompt
        self.reranker = MonoT5Reranker()
        self.stream_func = stream_func

    def ask(self, query: str, stream: bool = False, run_retrieve: bool = True, *args, **kwargs) -> tuple[
        str, List[Passage]]:
        decompose = QueryDecomposition(model_name=self.decompose_model_name, api_base=self.api_base)
        decompose_query: List[str] = decompose.decompose(query)
        is_decomposed = True
        if len(decompose_query) <= 0:
            is_decomposed = False

        if not run_retrieve and len(self.retrieved_passages) > 0:
            passage_list = self.retrieved_passages
        else:
            passage_list = []
            if is_decomposed:
                # use decomposed query
                for query in decompose_query:
                    hits = self.retrieval.retrieve(query, top_k=self.retrieve_size)
                    passage_list.extend(hits)
                passage_list = self.reranker.rerank(query, passage_list)
            else:
                hits = self.retrieval.retrieve(query, top_k=self.retrieve_size)
                passage_list.extend(hits)
            self.retrieved_passages = passage_list

        # remove duplicate elements while preserving order
        remove_duplicated = list(OrderedDict.fromkeys(passage_list))
        final_passages = remove_duplicated[:self.use_passage_count]
        input_prompt = deepcopy(self.prompt)
        for i, passage in enumerate(final_passages):
            input_prompt += f"[Document {i + 1}]: {passage.content}\n\n"
        input_prompt += f"Question: {query}\n\nAnswer: "

        answer = self.generate(input_prompt, self.model_name,
                               stream=stream,
                               stream_func=self.stream_func,
                               max_tokens=1024,
                               temperature=0.2,
                               *args, **kwargs)
        return answer, final_passages
