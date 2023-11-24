import pytest

from RAGchain.schema import RAGchainPromptTemplate, RAGchainChatPromptTemplate

normal_prompt_template = """
    Answer the question using below passages.
    Question: {question}

    Passages: {passages}
    """

no_passage_prompt_template = "Question: {question}"
no_question_prompt_template = "Answer the question using below passages.\n Passages: {passages}"
both_missing_prompt_template = "Answer the question using below passages.\nPassage: {passage}"


def test_ragchain_prompt_templates():
    normal_prompt = RAGchainPromptTemplate.from_template(normal_prompt_template)
    assert 'question' in normal_prompt.input_variables
    assert 'passages' in normal_prompt.input_variables

    with pytest.raises(ValueError):
        no_passage_prompt = RAGchainPromptTemplate.from_template(no_passage_prompt_template)

    with pytest.raises(ValueError):
        no_question_prompt = RAGchainPromptTemplate.from_template(no_question_prompt_template)

    with pytest.raises(ValueError):
        both_missing_prompt = RAGchainPromptTemplate.from_template(both_missing_prompt_template)


def test_ragchain_chat_prompt_templates():
    normal_prompt = RAGchainChatPromptTemplate.from_template(normal_prompt_template)
    assert 'question' in normal_prompt.input_variables
    assert 'passages' in normal_prompt.input_variables

    with pytest.raises(ValueError):
        no_passage_prompt = RAGchainChatPromptTemplate.from_template(no_passage_prompt_template)

    with pytest.raises(ValueError):
        no_question_prompt = RAGchainChatPromptTemplate.from_template(no_question_prompt_template)

    with pytest.raises(ValueError):
        both_missing_prompt = RAGchainChatPromptTemplate.from_template(both_missing_prompt_template)
