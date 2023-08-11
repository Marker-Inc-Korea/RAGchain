from typing import List

import pandas as pd
from huggingface_hub import hf_hub_download
from langchain.schema import Document

from KoPrivateGPT.preprocess.loader.base import BaseLoader


class KoStrategyQALoader(BaseLoader):
    """
        KoStrategyQA dataset loader
        The dataset downloads at huggingface via internet.
    """

    def __init__(self, *args, **kwargs):
        paragraph_path = hf_hub_download(repo_id="NomaDamas/Ko-StrategyQA",
                                         filename="ko-strategy-qa_paragraphs.parquet",
                                         repo_type="dataset")
        self.paragraph_df = pd.read_parquet(paragraph_path)

    def load(self) -> List[Document]:
        def make_document(row):
            return Document(page_content=row["ko-content"], metadata={"id": row["key"]})

        self.paragraph_df["document"] = self.paragraph_df.apply(make_document, axis=1)
        return self.paragraph_df["document"].tolist()
