import pytest
from datasets import load_dataset
from langchain.chat_models import ChatOpenAI
from langchain.globals import set_debug
from langchain.memory import ChatMessageHistory
from langchain.schema import StrOutputParser
from langchain.schema.runnable.history import RunnableWithMessageHistory

from RAGchain.utils.search_detector import SearchDetector


@pytest.fixture
def dstc11_track5_dataset():
    dataset_repo = "NomaDamas/DSTC-11-Track-5"
    test_dataset = load_dataset(dataset_repo, split='test')
    return test_dataset.to_pandas()


def logs_to_history(logs):
    history = ChatMessageHistory()
    for log in logs:
        if log['speaker'] == 'U':
            history.add_user_message(log['text'])
        elif log['speaker'] == 'S':
            history.add_ai_message(log['text'])
        else:
            raise ValueError(f"Unknown speaker: {log['speaker']}")
    return history


def decide_answer(answer: str) -> bool:
    answer = answer.lower()
    if 'yes' in answer:
        return True
    elif 'no' in answer:
        return False
    else:
        return False  # select False as default...


def main():
    dataset_repo = "NomaDamas/DSTC-11-Track-5"
    test_dataset = load_dataset(dataset_repo, split='test')
    test_df = test_dataset.to_pandas()
    test_df = test_df.sample(500, random_state=42)
    test_df['history'] = test_df.apply(lambda row: logs_to_history(row['log']), axis=1)

    detector = SearchDetector()
    runnable = detector.prompt | ChatOpenAI(max_tokens=8) | StrOutputParser()
    chain_with_history = RunnableWithMessageHistory(runnable,
                                                    lambda row_index: test_df['history'].iloc[row_index],
                                                    history_messages_key="history")
    answers = chain_with_history.batch(
        inputs=[{} for _ in range(len(test_df))],
        config=[{"configurable": {"session_id": row_index}} for row_index in range(len(test_df))]
    )

    test_df['answer'] = answers
    test_df['prediction'] = test_df.apply(lambda row: decide_answer(row['answer']), axis=1)

    from sklearn.metrics import classification_report
    report = classification_report(test_df['target'], test_df['prediction'])
    print(report)

    test_df.to_csv('./search_detector_test_few_shot.csv', index=False)


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    set_debug(False)
    main()
