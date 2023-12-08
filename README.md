# RAGchain

RAGchain is a framework for developing advanced RAG(Retrieval Augmented Generation) workflow powered by LLM (Large Language Model).
While existing frameworks like Langchain or LlamaIndex allow you to build simple RAG workflows, they have limitations when it comes to building complex and high-accuracy RAG workflows.

RAGchain is designed to overcome these limitations by providing powerful features for building advanced RAG workflow easily.
Also, it is partially compatible with Langchain, allowing you to leverage many of its integrations for vector storage,
embeddings, document loaders, and LLM models.

[Docs](https://nomadamas.gitbook.io/ragchain-docs/) | [API Spec](https://nomadamas.github.io/RAGchain) | [QuickStart](https://nomadamas.gitbook.io/ragchain-docs/quick-start)

# Quick Install
```bash
pip install RAGchain
```

# Why RAGchain?
RAGchain offers several powerful features for building high-quality RAG workflows:

## OCR Loaders
Simple file loaders may not be sufficient when trying to enhance accuracy or ingest real-world documents. OCR models can scan documents and convert them into text with high accuracy, improving the quality of responses from LLMs.

## Reranker
Reranking is a popular method used in many research projects to improve retrieval accuracy in RAG workflows. Unlike LangChain, which doesn't include reranking as a default feature, RAGChain comes with various rerankers.

## Great to use multiple retrievers
In real-world scenarios, you may need multiple retrievers depending on your requirements. RAGchain is highly optimized for using multiple retrievers. It divides retrieval and DB. Retrieval saves vector representation of contents, and DB saves contents. We connect both with Linker, so it is really easy to use multiple retrievers and DBs.

## pre-made RAG pipelines
We provide pre-made pipelines that let you quickly set up RAG workflow. We are planning to make much complex pipelines, which hard to make but powerful. With pipelines, you can build really powerful RAG system quickly and easily. 

## Easy benchmarking

It is crucial to benchmark and test your RAG workflows. We have easy benchmarking module for evaluation. Support your
own questions and various datasets.


# Installation
## From pip

simply install at pypi.

```bash
pip install RAGchain
```

## From source
First, clone this git repository to your local machine.

```bash
git clone https://github.com/NomaDamas/RAGchain.git
cd RAGchain
```

Then, install RAGchain module.
```bash
python3 setup.py develop
```

For using files at root folder and test, run dev requirements.
```bash
pip install dev_requirements.txt
```

# Supporting Features
## Retrievals
- BM25
- Vector DB
- Hybrid ([rrf](https://www.elastic.co/guide/en/elasticsearch/reference/current/rrf.html) and [cc](https://arxiv.org/abs/2210.11934))
- [HyDE](https://arxiv.org/abs/2212.10496)

## OCR Loaders

- [Nougat](https://github.com/facebookresearch/nougat)
- [Deepdoctection](https://github.com/deepdoctection/deepdoctection)

## Rerankers
- [UPR](https://github.com/DevSinghSachan/unsupervised-passage-reranking)
- [TART](https://github.com/facebookresearch/tart)
- BM25
- LLM
- [MonoT5](https://huggingface.co/castorini/monot5-3b-msmarco-10k)

## Workflows (pipeline or LLM)
- Basic
- [Visconde](https://arxiv.org/abs/2212.09656)

## Extra utils
- Query Decomposition
- Evidence Extractor

## Dataset Evaluators

- [MS-MARCO](https://paperswithcode.com/dataset/ms-marco)
- [Mr. Tydi](https://arxiv.org/abs/2108.08787)
- [Qasper](https://paperswithcode.com/dataset/qasper)
- [StrategyQA](https://allenai.org/data/strategyqa)
- [KoStrategyQA](https://huggingface.co/datasets/NomaDamas/Ko-StrategyQA)

# Contributing
We welcome any contributions. Please feel free to raise issues and submit pull requests.

# Acknowledgement
This project was developed by [NomaDamas](https://github.com/NomaDamas), an open-source project group based in Seoul. The project is licensed under the Apache 2.0 License.
