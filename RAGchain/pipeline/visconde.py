from typing import List

from langchain.llms import BaseLLM
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnableLambda

from RAGchain.pipeline.base import BaseRunPipeline
from RAGchain.reranker import MonoT5Reranker
from RAGchain.retrieval.base import BaseRetrieval
from RAGchain.schema import Passage, RAGchainPromptTemplate, RetrievalResult
from RAGchain.utils.query_decompose import QueryDecomposition


class ViscondeRunPipeline(BaseRunPipeline):
    strategyqa_prompt = RAGchainPromptTemplate.from_template("""For each example, use the documents to create an \"Answer\" and an \"Explanation\" to the \"Question\". Just answer yes or no.

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

        Example 2:
        
        {passages}
        
        Question: {question}
        Answer:
        """)

    def __init__(self,
                 retrieval: BaseRetrieval,
                 llm: BaseLLM,
                 decompose: QueryDecomposition = None,
                 prompt: RAGchainPromptTemplate = None,
                 use_passage_count: int = 3,
                 ):
        """
        Initializes an instance of the ViscondeRunPipeline class.

        :param retrieval: An instance of the Retrieval module used for retrieving passages.
        :param llm: An instance of the LLM module used for answering questions. You can't use chat models for Visconde Pipeline.
        :param decompose: An instance of the QueryDecomposition module used for decomposing query. Default is QueryDecomposition().
        :param prompt: RAGchainPromptTemplate used for generating prompts based on passages and user query.
        Default is ViscondeRunPipeline.strategyqa_prompt.
        :param use_passage_count: The number of passages to be used for llm question answering. Default is 3.
        """
        self.retrieval = retrieval
        self.llm = llm
        self.decompose = decompose if decompose is not None else QueryDecomposition(self.llm)
        self.prompt = prompt if prompt is not None else self.strategyqa_prompt
        self.reranker = MonoT5Reranker()
        self.use_passage_count = use_passage_count
        super().__init__()

    def _make_runnable(self):
        self.run = self.decompose | RunnableLambda(
            lambda queries: sum(self.retrieval.batch(
                queries, config={'configurable': {"retrieval_options": {"top_k": 50}}}
            ))  # sum can drop duplicate elements automatically
        ) | self.reranker | RunnableLambda(
            lambda x: x.slice(end=self.use_passage_count).to_prompt_input()
        ) | self.prompt | self.llm | StrOutputParser()

    def get_passages_and_run(self, questions: List[str], top_k: int = 50) -> tuple[
        List[str], List[List[Passage]], List[List[float]]]:
        runnable = self.decompose | RunnableLambda(
            lambda queries: sum(self.retrieval.batch(
                queries, config={'configurable': {"retrieval_options": {"top_k": top_k}}}
            ))
        ) | self.reranker | RunnableLambda(
            lambda x: x.slice(end=self.use_passage_count)
        ) | {
                       "passages": RunnableLambda(lambda x: x.passages),
                       "scores": RunnableLambda(lambda x: x.scores),
                       "answer": RunnableLambda(
                           RetrievalResult.to_prompt_input) | self.prompt | self.llm | StrOutputParser()
                   }
        results = runnable.batch(questions)
        answers, passages, rel_scores = zip(
            *[(result['answer'], result['passages'], result['scores']) for result in results])
        return list(answers), list(passages), list(rel_scores)
