"""
This code is from pygaggle.
https://github.com/castorini/pygaggle/blob/master/pygaggle/rerank/transformer.py
"""
from copy import deepcopy
from typing import List

import torch
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          T5ForConditionalGeneration)

from model import greedy_decode, QueryDocumentBatchTokenizer, QueryDocumentBatch, T5BatchTokenizer
from .base import Query, Text, Reranker

__all__ = ['MonoT5']

prediction_tokens = {
    'castorini/monot5-base-msmarco': ['▁false', '▁true'],
    'castorini/monot5-base-msmarco-10k': ['▁false', '▁true'],
    'castorini/monot5-large-msmarco': ['▁false', '▁true'],
    'castorini/monot5-large-msmarco-10k': ['▁false', '▁true'],
    'castorini/monot5-base-med-msmarco': ['▁false', '▁true'],
    'castorini/monot5-3b-med-msmarco': ['▁false', '▁true'],
    'castorini/monot5-3b-msmarco-10k': ['▁false', '▁true'],
    'unicamp-dl/mt5-base-en-msmarco': ['▁no', '▁yes'],
    'unicamp-dl/ptt5-base-pt-msmarco-10k-v2': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-pt-msmarco-100k-v2': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2': ['▁não', '▁sim'],
    'unicamp-dl/mt5-base-en-pt-msmarco-v2': ['▁no', '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v2': ['▁no', '▁yes'],
    'unicamp-dl/mt5-base-en-pt-msmarco-v1': ['▁no', '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v1': ['▁no', '▁yes'],
    'unicamp-dl/ptt5-base-pt-msmarco-10k-v1': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-pt-msmarco-100k-v1': ['▁não', '▁sim'],
    'unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1': ['▁não', '▁sim'],
    'unicamp-dl/mt5-3B-mmarco-en-pt': ['▁', '▁true'],
    'unicamp-dl/mt5-13b-mmarco-100k': ['▁', '▁true'],
}


class MonoT5(Reranker):
    def __init__(self,
                 pretrained_model_name_or_path: str = 'castorini/monot5-base-msmarco-10k',
                 model: T5ForConditionalGeneration = None,
                 tokenizer: QueryDocumentBatchTokenizer = None,
                 use_amp=False,
                 token_false=None,
                 token_true=None):
        self.model = model or self.get_model(pretrained_model_name_or_path)
        self.tokenizer = tokenizer or self.get_tokenizer(pretrained_model_name_or_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
            pretrained_model_name_or_path, self.tokenizer, token_false, token_true)
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.device = next(self.model.parameters(), None).device
        self.use_amp = use_amp

    @staticmethod
    def get_model(pretrained_model_name_or_path: str,
                  *args, device: str = None, **kwargs) -> T5ForConditionalGeneration:
        device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device(device)
        return AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path,
                                                     *args, **kwargs).to(device).eval()

    @staticmethod
    def get_tokenizer(pretrained_model_name_or_path: str,
                      *args, batch_size: int = 8, **kwargs) -> T5BatchTokenizer:
        return T5BatchTokenizer(
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False, *args, **kwargs),
            batch_size=batch_size
        )

    @staticmethod
    def get_prediction_tokens(pretrained_model_name_or_path: str,
                              tokenizer, token_false, token_true):
        if not (token_false and token_true):
            if pretrained_model_name_or_path in prediction_tokens:
                token_false, token_true = prediction_tokens[pretrained_model_name_or_path]
                token_false_id = tokenizer.tokenizer.get_vocab()[token_false]
                token_true_id = tokenizer.tokenizer.get_vocab()[token_true]
                return token_false_id, token_true_id
            else:
                raise Exception(f"We don't know the indexes for the non-relevant/relevant tokens for\
                        the checkpoint {pretrained_model_name_or_path} and you did not provide any.")
        else:
            token_false_id = tokenizer.tokenizer.get_vocab()[token_false]
            token_true_id = tokenizer.tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id

    def rescore(self, query: Query, texts: List[Text]) -> List[Text]:
        texts = deepcopy(texts)
        batch_input = QueryDocumentBatch(query=query, documents=texts)
        for batch in self.tokenizer.traverse_query_document(batch_input):
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                input_ids = batch.output['input_ids'].to(self.device)
                attn_mask = batch.output['attention_mask'].to(self.device)
                _, batch_scores = greedy_decode(self.model,
                                                input_ids,
                                                length=1,
                                                attention_mask=attn_mask,
                                                return_last_logits=True)

                batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
                batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
                batch_log_probs = batch_scores[:, 1].tolist()
            for doc, score in zip(batch.documents, batch_log_probs):
                doc.score = score

        return texts
