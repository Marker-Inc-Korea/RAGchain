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
from typing import List

import openai

from KoPrivateGPT.DB.base import BaseDB
from KoPrivateGPT.llm.base import BaseLLM
from KoPrivateGPT.llm.basic import BasicLLM
from KoPrivateGPT.retrieval.base import BaseRetrieval
from KoPrivateGPT.schema import Passage
from KoPrivateGPT.utils import QueryDecomposition
from KoPrivateGPT.utils.reranker import MonoT5Reranker


class ViscondeLLM(BaseLLM):
    qasper_prompt = """For each example, explain how each document is used to answer the question:
    
    Example 1:
    
    [Document 1]: In this section we describe a number of experiments targeted to compare the performance of popular named entity recognition algorithms on our data. We trained and evaluated Stanford NER, spaCy 2.0, and a recurrent model similar to BIBREF13 , BIBREF14 that uses bidirectional LSTM cells for character-based feature extraction and CRF, described in Guillaume Genthial's Sequence Tagging with Tensorflow blog post BIBREF15 .
    
    [Document 2]: Stanford NER is conditional random fields (CRF) classifier based on lexical and contextual features such as the current word, character-level n-grams of up to length 6 at its beginning and the end, previous and next words, word shape and sequence features BIBREF16 .
    
    [Document 3]: spaCy 2.0 uses a CNN-based transition system for named entity recognition. For each token, a Bloom embedding is calculated based on its lowercase form, prefix, suffix and shape, then using residual CNNs, a contextual representation of that token is extracted that potentially draws information from up to 4 tokens from each side BIBREF17 . Each update of the transition system's configuration is a classification task that uses the contextual representation of the top token on the stack, preceding and succeeding tokens, first two tokens of the buffer, and their leftmost, second leftmost, rightmost, second rightmost children. The valid transition with the highest score is applied to the system. This approach reportedly performs within 1% of the current state-of-the-art for English . In our experiments, we tried out 50-, 100-, 200- and 300-dimensional pre-trained GloVe embeddings. Due to time constraints, we did not tune the rest of hyperparameters and used their default values.
    
    [Document 4]: In order to evaluate the models trained on generated data, we manually annotated a named entities dataset comprising 53453 tokens and 2566 sentences selected from over 250 news texts from ilur.am. This dataset is comparable in size with the test sets of other languages (Table TABREF10 ). Included sentences are from political, sports, local and world news (Figures FIGREF8 , FIGREF9 ), covering the period between August 2012 and July 2018. The dataset provides annotations for 3 popular named entity classes: people (PER), organizations (ORG), and locations (LOC), and is released in CoNLL03 format with IOB tagging scheme. Tokens and sentences were segmented according to the UD standards for the Armenian language BIBREF11 .
    
    [Document 5]: The main model that we focused on was the recurrent model with a CRF top layer, and the above-mentioned methods served mostly as baselines. The distinctive feature of this approach is the way contextual word embeddings are formed. For each token separately, to capture its word shape features, character-based representation is extracted using a bidirectional LSTM BIBREF18 . This representation gets concatenated with a distributional word vector such as GloVe, forming an intermediate word embedding. Using another bidirectional LSTM cell on these intermediate word embeddings, the contextual representation of tokens is obtained (Figure FIGREF17 ). Finally, a CRF layer labels the sequence of these contextual representations. In our experiments, we used Guillaume Genthial's implementation of the algorithm. We set the size of character-based biLSTM to 100 and the size of second biLSTM network to 300
    
    Question: what ner models were evaluated?
    
    Answer: Stanford NER algorithm, the spaCy 2.0 algorithm, recurrent model with a CRF top layer.
    
    Example 2:
    
    """

    def __init__(self,
                 retrieval: BaseRetrieval,
                 db: BaseDB,
                 model_name: str = "text-davinci-003",
                 api_base: str = None,
                 decompose_model_name: str = "text-davinci-003",
                 retrieve_size: int = 50,
                 use_passage_count: int = 3,
                 prompt: str = None,
                 *args, **kwargs):
        self.retrieval = retrieval
        self.db = db
        self.model_name = model_name
        self.decompose_model_name = decompose_model_name
        BasicLLM.set_model(api_base)
        self.api_base = api_base
        self.retrieve_size = retrieve_size
        self.use_passage_count = use_passage_count
        if prompt is not None:
            self.prompt = prompt
        else:
            self.prompt = self.qasper_prompt
        self.reranker = MonoT5Reranker()

    def ask(self, query: str) -> tuple[str, List[Passage]]:
        decompose = QueryDecomposition(model_name=self.decompose_model_name, api_base=self.api_base)
        decompose_query: List[str] = decompose.decompose(query)
        is_decomposed = True
        if len(decompose_query) <= 0:
            is_decomposed = False

        passage_list = []
        if is_decomposed:
            # use decomposed query
            for query in decompose_query:
                hits = self.retrieval.retrieve(query, self.db, top_k=self.retrieve_size)
                passage_list.extend(hits)
            passage_list = self.reranker.rerank(query, passage_list)
        else:
            hits = self.retrieval.retrieve(query, self.db, top_k=self.retrieve_size)
            passage_list.extend(hits)

        # remove duplicate elements while preserving order
        remove_duplicated = list(OrderedDict.fromkeys(passage_list))
        final_passages = remove_duplicated[:self.use_passage_count]
        input_prompt = deepcopy(self.prompt)
        for i, passage in enumerate(final_passages):
            input_prompt += f"[Document {i + 1}]: {passage.content}\n\n"
        input_prompt += f"Question: {query}\n\nAnswer: "

        answer = self.generate(input_prompt)
        return answer, final_passages

    def generate(self, prompt: str, max_tokens=1024, temperature=0):
        response = openai.Completion.create(
            model=self.model_name,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["choices"][0]["text"]
