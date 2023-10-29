from collections import OrderedDict
from typing import Callable, List

from RAGchain.llm.completion import CompletionLLM
from RAGchain.pipeline.base import BasePipeline
from RAGchain.reranker import MonoT5Reranker
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage
from RAGchain.utils.query_decompose import QueryDecomposition


class ViscondeRunPipeline(BasePipeline):
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
                 decompose: QueryDecomposition = None,
                 prompt_func: Callable[[List[Passage], str], str] = None,
                 *args, **kwargs
                 ):
        """
        Initializes an instance of the ViscondeRunPipeline class.

        :param retrieval: An instance of the Retrieval module used for retrieving passages.
        :param decompose: An instance of the QueryDecomposition module used for decomposing query. Default is QueryDecomposition().
        :param prompt_func: A callable function used for generating prompts based on passages and user query. The input of prompt_func will be the list of retrieved passages and user query. The output of prompt_func should be a string. Default is ViscondeRunPipeline.default_prompt.
        :param args: optional parameter for initialization of CompletionLLM
        :param kwargs: optional parameter for initialization of CompletionLLM
        """
        self.retrieval = retrieval
        self.decompose = decompose if decompose is not None else QueryDecomposition()
        self.prompt_func = prompt_func if prompt_func is not None else self.default_prompt
        self.llm = CompletionLLM(prompt_func=self.prompt_func, *args, **kwargs)
        self.reranker = MonoT5Reranker()

    def run(self,
            query: str,
            retrieve_size: int = 50,
            use_passage_count: int = 3,
            *args, **kwargs):
        """
        :param query: question
        :param retrieve_size: The number of passages to be retrieved before reranking. Default is 50.
        :param use_passage_count: The number of passages to be used for llm question answering. Default is 3.
        :param args: optional parameter for llm.ask()
        :param kwargs: optional parameter for llm.ask()
        """
        decompose_query: List[str] = self.decompose.decompose(query)
        passage_list = []
        if len(decompose_query) > 0:
            # use decomposed query
            for query in decompose_query:
                hits = self.retrieval.retrieve(query, top_k=retrieve_size)
                passage_list.extend(hits)
            passage_list = self.reranker.rerank(query, passage_list)
        else:
            hits = self.retrieval.retrieve(query, top_k=retrieve_size)
            passage_list.extend(hits)

        # remove duplicate elements while preserving order
        remove_duplicated = list(OrderedDict.fromkeys(passage_list))
        final_passages = remove_duplicated[:use_passage_count]

        return self.llm.ask(query, final_passages, *args, **kwargs)

    def default_prompt(self, passages: List[Passage], question: str):
        passage_str = "\n\n".join([f"[Document {i + 1}]: {passage.content}" for i, passage in enumerate(passages)])
        return f"""{self.strategyqa_prompt}
        {passage_str}
        
        Question: {question}
        Answer:
        """
