import pytest

from RAGchain.schema import RAGchainPromptTemplate


def test_ragchain_prompt_templates():
    normal_prompt = RAGchainPromptTemplate.from_template(
        """
        Answer the question using below passages.
        Question: {question}

        Passages: {passages}
        """
    )
    assert 'question' in normal_prompt.input_variables
    assert 'passages' in normal_prompt.input_variables

    with pytest.raises(ValueError):
        no_passage_prompt = RAGchainPromptTemplate.from_template(
            """
            Question: {question}
            """
        )

    with pytest.raises(ValueError):
        no_question_prompt = RAGchainPromptTemplate.from_template(
            """
            Answer the question using below passages.

            Passages: {passages}
            """
        )

    with pytest.raises(ValueError):
        both_missing_prompt = RAGchainPromptTemplate.from_template(
            """
            Answer the question using below passages.

            Passage: {passage}
            """
        )
