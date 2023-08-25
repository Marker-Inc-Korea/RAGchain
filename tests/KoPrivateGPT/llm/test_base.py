from uuid import UUID

from KoPrivateGPT.schema import Passage

TEST_PASSAGES = [
        Passage(id=UUID('d2297ab3-ded2-43b3-84e9-bb512f17f733'),
                content='base-dot-v5.\n\n\x0cstep, we applied four methods: 1) using GPT-3 without CoT and providing '
                        'the\nlinks and the ground truth contexts, i.e., skipping the document retrieval step;\n2) using '
                        'the reasoning step with the links and ground truth contexts; 3) using\nreasoning step over the '
                        'intersection of retrieved documents and the documents\ncited by the main context; and 4) '
                        'reasoning over the documents retrieved from\nthe entire Wikipedia subset provided by the '
                        'dataset.\n\n4.2 Qasper',
                filepath='/Users/jeffrey/PycharmProjects/KoPrivateGPT/SOURCE_DOCUMENTS/Visconde.pdf',
                previous_passage_id=None,
                next_passage_id=UUID('ef85ba2f-4325-43e3-9197-c0d57135b795'), metadata_etc={}),
        Passage(id=UUID('ef85ba2f-4325-43e3-9197-c0d57135b795'),
                content='4.2 Qasper\n\nQasper [5] is an information-seeking QA dataset over academic research '
                        'papers.\nThe task consists of retrieving the most relevant evidence paragraph for each\nquestion '
                        'and answering the question.',
                filepath='/Users/jeffrey/PycharmProjects/KoPrivateGPT/SOURCE_DOCUMENTS/Visconde.pdf',
                previous_passage_id=UUID('d2297ab3-ded2-43b3-84e9-bb512f17f733'),
                next_passage_id=UUID('f94887b6-91f4-4b68-a29e-5fe1a278ccdf'), metadata_etc={}),
        Passage(id=UUID('f94887b6-91f4-4b68-a29e-5fe1a278ccdf'),
                content='Procedure: we did not apply query decomposition because the questions in\nthis dataset are '
                        'closed-ended, i.e., they do not require decomposition as they are\ngrounded in a single paper of '
                        'interest [5]. For example, the question “How is the\ntext segmented?” only makes sense '
                        'concerning its grounded paper. Besides, we\nskipped the BM25 step for document retrieval as the '
                        'monoT5 reranker can score\neach paragraph in the paper in a reasonable time. The document '
                        'retrieval step',
                filepath='/Users/jeffrey/PycharmProjects/KoPrivateGPT/SOURCE_DOCUMENTS/Visconde.pdf',
                previous_passage_id=UUID('ef85ba2f-4325-43e3-9197-c0d57135b795'),
                next_passage_id=UUID('27b4c526-7f92-4b75-b783-fae5a9bda437'), metadata_etc={}),
        Passage(id=UUID('27b4c526-7f92-4b75-b783-fae5a9bda437'),
                content='consists of reranking the paper’s paragraphs based on the question and choosing\nthe top ﬁve as '
                        'context documents. We did not notice any advantage in using a\ndynamic prompt in this dataset.',
                filepath='/Users/jeffrey/PycharmProjects/KoPrivateGPT/SOURCE_DOCUMENTS/Visconde.pdf',
                previous_passage_id=UUID('f94887b6-91f4-4b68-a29e-5fe1a278ccdf'),
                next_passage_id=UUID('a6cfd8ca-682d-4abb-b6ef-33726e5c2be6'), metadata_etc={}),
        Passage(id=UUID('a6cfd8ca-682d-4abb-b6ef-33726e5c2be6'), content='4.3 StrategyQA',
                filepath='/Users/jeffrey/PycharmProjects/KoPrivateGPT/SOURCE_DOCUMENTS/Visconde.pdf',
                previous_passage_id=UUID('27b4c526-7f92-4b75-b783-fae5a9bda437'),
                next_passage_id=UUID('94a50e43-08eb-4e66-8ab6-fe3702dda8c6'), metadata_etc={}),
        Passage(id=UUID('94a50e43-08eb-4e66-8ab6-fe3702dda8c6'),
                content='StrategyQA [9] is a dataset focused on open-domain questions that require reason-\ning steps. '
                        'This dataset has three tasks: 1) question decomposition, measured using\na metric called SARI, '
                        'generally used to evaluate automatic text simpliﬁcation\nsystems [35]; 2) evidence paragraph '
                        'retrieval, measured as the recall of the top\nten retrieved results; and 3) question answering, '
                        'measured in terms of accuracy.\nPre-processing: we did not generate reasoning paragraphs for the '
                        'training',
                filepath='/Users/jeffrey/PycharmProjects/KoPrivateGPT/SOURCE_DOCUMENTS/Visconde.pdf',
                previous_passage_id=UUID('a6cfd8ca-682d-4abb-b6ef-33726e5c2be6'),
                next_passage_id=UUID('0fa38fe0-6aca-4fcd-aff2-6274cbfef33f'), metadata_etc={}),
        Passage(id=UUID('0fa38fe0-6aca-4fcd-aff2-6274cbfef33f'),
                content='examples since the context comprises long paragraphs that exceed the model\ninput size limit ('
                        '4000 tokens). We processed the context articles provided by the\ndataset to create a searchable '
                        'index, by splitting the articles into windows of\nthree sentences each.',
                filepath='/Users/jeffrey/PycharmProjects/KoPrivateGPT/SOURCE_DOCUMENTS/Visconde.pdf',
                previous_passage_id=UUID('94a50e43-08eb-4e66-8ab6-fe3702dda8c6'),
                next_passage_id=UUID('9d5c08b0-6904-4321-b5e6-880cd0f5d4e6'), metadata_etc={}),
        Passage(id=UUID('9d5c08b0-6904-4321-b5e6-880cd0f5d4e6'),
                content='Procedure: we applied question decomposition and performed retrieval using\nthe approach '
                        'described in Section 3. We used the top ﬁve retrieved documents\nfor each decomposed question as '
                        'the context in the reading step.\n\n4.4 Results',
                filepath='/Users/jeffrey/PycharmProjects/KoPrivateGPT/SOURCE_DOCUMENTS/Visconde.pdf',
                previous_passage_id=UUID('0fa38fe0-6aca-4fcd-aff2-6274cbfef33f'),
                next_passage_id=UUID('d10d22d2-1b3b-4aa2-ba8c-0610504c3044'), metadata_etc={}),
        Passage(id=UUID('d10d22d2-1b3b-4aa2-ba8c-0610504c3044'),
                content='4.4 Results\n\nIn Table 1, we present the results of our experiments. First, we show the '
                        'results\nobtained in the IIRC dataset. Our approach outperforms the baselines (i.e.,\nFerguson '
                        'et al. [7]’s) in diﬀerent settings: 1) Using the gold context searched by\nhumans (Gold Ctx); 2) '
                        'Searching for context in the links the dataset provides\n(Linked pages); and 3) Searching for '
                        'contexts in the entire dataset. We report\n\n\x0cTable 1: Visconde and similar methods results '
                        'on IIRC, Qasper and StrategyQA.\n\nIIRC',
                filepath='/Users/jeffrey/PycharmProjects/KoPrivateGPT/SOURCE_DOCUMENTS/Visconde.pdf',
                previous_passage_id=UUID('9d5c08b0-6904-4321-b5e6-880cd0f5d4e6'),
                next_passage_id=None, metadata_etc={})
]
