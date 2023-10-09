from typing import List

try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError:
    raise ImportError("Please install transformers and torch")

from KoPrivateGPT.reranker.base import BaseReranker
from KoPrivateGPT.schema import Passage


class UPRReranker(BaseReranker):
    def __init__(self,
                 model_name: str = "t5-large",
                 prefix_prompt: str = "Passage: ",
                 suffix_prompt: str = "Please write a question based on this passage.",
                 use_bf16: bool = False,
                 use_gpu: bool = False,
                 shard_size: int = 16):
        self.prefix_prompt = prefix_prompt
        self.suffix_prompt = suffix_prompt
        self.model = T5ForConditionalGeneration.from_pretrained(model_name,
                                                                torch_dtype=torch.bfloat16 if use_bf16 else torch.float32)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.use_gpu = use_gpu
        self.shard_size = shard_size

    def rerank(self, query: str, passages: List[Passage]) -> List[Passage]:
        input_contexts = [f"{passage.filepath} {passage.content}" for passage in passages]
        indexes, _ = self.calculate_likelihood(query, input_contexts)
        reranked_passages = [passages[idx] for idx in indexes]
        return reranked_passages

    def rerank_sliding_window(self, query: str, passages: List[Passage], window_size: int) -> List[Passage]:
        raise NotImplementedError("UPR reranker does not support sliding window reranking.")

    def calculate_likelihood(self, question: str, contexts: List[str]) -> tuple[List[int], List[float]]:
        prompts = [f"{self.prefix_prompt} {context} {self.suffix_prompt}" for context in contexts]
        # tokenize contexts and instruction prompts
        context_tokens = self.tokenizer(prompts,
                                        padding='longest',
                                        max_length=512,
                                        pad_to_multiple_of=8,
                                        truncation=True,
                                        return_tensors='pt')
        context_tensor, context_attention_mask = context_tokens.input_ids, context_tokens.attention_mask
        if self.use_gpu:
            context_tensor, context_attention_mask = context_tensor.cuda(), context_attention_mask.cuda()

        # tokenize question
        question_tokens = self.tokenizer([question],
                                         max_length=128,
                                         truncation=True,
                                         return_tensors='pt')
        question_tensor = question_tokens.input_ids
        if self.use_gpu:
            question_tensor = question_tensor.cuda()
        question_tensor = torch.repeat_interleave(question_tensor, len(contexts), dim=0)

        sharded_nll_list = []

        # calculate log likelihood
        for i in range(0, len(context_tensor), self.shard_size):
            encoder_tensor_view = context_tensor[i: i + self.shard_size]
            attention_mask_view = context_attention_mask[i: i + self.shard_size]
            decoder_tensor_view = question_tensor[i: i + self.shard_size]
            with torch.no_grad():
                logits = self.model(input_ids=encoder_tensor_view,
                                    attention_mask=attention_mask_view,
                                    labels=decoder_tensor_view).logits

            log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)
            nll = -log_softmax.gather(2, decoder_tensor_view.unsqueeze(2)).squeeze(2)

            avg_nll = torch.sum(nll, dim=1)
            sharded_nll_list.append(avg_nll)

        topk_scores, indexes = torch.topk(-torch.cat(sharded_nll_list), k=len(context_tensor))

        return indexes.tolist(), topk_scores.tolist()
