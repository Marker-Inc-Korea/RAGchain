import os

from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings

from RAGchain.utils.embed import EmbeddingFactory


def test_embedding_factory():
    openai_embedding = EmbeddingFactory(embed_type='openai').get()
    assert isinstance(openai_embedding, OpenAIEmbeddings)
    assert openai_embedding.openai_api_key == os.getenv('OPENAI_API_KEY')

    kosimcse_embedding = EmbeddingFactory(embed_type='kosimcse').get()
    assert isinstance(kosimcse_embedding, HuggingFaceEmbeddings)
    assert kosimcse_embedding.model_name == "BM-K/KoSimCSE-roberta-multitask"

    ko_sroberta_multitask_embedding = EmbeddingFactory(embed_type='ko_sroberta_multitask').get()
    assert isinstance(ko_sroberta_multitask_embedding, HuggingFaceEmbeddings)
    assert ko_sroberta_multitask_embedding.model_name == "jhgan/ko-sroberta-multitask"

    multilingual_e5_embedding = EmbeddingFactory(embed_type='multilingual_e5').get()
    assert isinstance(multilingual_e5_embedding, HuggingFaceEmbeddings)
    assert multilingual_e5_embedding.model_name == "intfloat/multilingual-e5-large"

    contriever_embedding = EmbeddingFactory(embed_type='contriever').get()
    assert isinstance(contriever_embedding, HuggingFaceEmbeddings)
    assert contriever_embedding.model_name == "facebook/mcontriever-msmarco"
