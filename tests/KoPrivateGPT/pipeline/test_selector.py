from KoPrivateGPT.pipeline.selector import ModuleSelector


def test_module_selector():
    assert ModuleSelector("file_loader").select("file_loader").module.__name__ == "FileLoader"
    assert ModuleSelector("file_loader").select("ko_strategy_qa_loader").module.__name__ == "KoStrategyQALoader"

    assert ModuleSelector("text_splitter").select("recursive_text_splitter").module.__name__ == "RecursiveTextSplitter"

    assert ModuleSelector("db").select("pickle_db").module.__name__ == "PickleDB"
    assert ModuleSelector("db").select("mongo_db").module.__name__ == "MongoDB"

    assert ModuleSelector("retrieval").select("bm25").module.__name__ == "BM25Retrieval"
    assert ModuleSelector("retrieval").select("vector_db").module.__name__ == "VectorDBRetrieval"
    assert ModuleSelector("retrieval").select("hyde").module.__name__ == "HyDERetrieval"

    assert ModuleSelector("llm").select("basic_llm").module.__name__ == "BasicLLM"
    assert ModuleSelector("llm").select("rerank_llm").module.__name__ == "RerankLLM"
    assert ModuleSelector("llm").select("visconde_llm").module.__name__ == "ViscondeLLM"
