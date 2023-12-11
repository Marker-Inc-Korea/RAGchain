from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture

load_dotenv()
from RAGchain.utils.rede_search_detector import RedeSearchDetector


def extract_last_log_text(row):
    return row.log[-1]['text']


def main():
    knowledge_turn_cnt = 2
    non_knowledge_turn_cnt = 50

    dataset_repo = "NomaDamas/DSTC-11-Track-5"
    train_dataset = load_dataset(dataset_repo, split='train')
    train_df = train_dataset.to_pandas()
    train_df = train_df.sample(200, random_state=42)

    knowledge_seeking_turns = train_df[train_df['target']]
    non_knowledge_seeking_turns = train_df[train_df['target'] == False]

    # slice for testing
    knowledge_seeking_turns = knowledge_seeking_turns[:knowledge_turn_cnt]
    non_knowledge_seeking_turns = non_knowledge_seeking_turns[:non_knowledge_turn_cnt]

    rede_detector = RedeSearchDetector(threshold=0.5)
    rede_detector.find_representation_transform([
        row.log[-1]['text'] for i, row in knowledge_seeking_turns.iterrows()
    ])
    rede_detector.train_density_estimation(
        GaussianMixture(n_components=1),
        [row.log[-1]['text'] for i, row in non_knowledge_seeking_turns.iterrows()])
    validation_dataset = load_dataset(dataset_repo, split='validation')
    validation_df = validation_dataset.to_pandas()
    validation_df = validation_df.sample(200, random_state=42)
    validation_knowledge_seeking_turns = validation_df[validation_df['target']][:int(knowledge_turn_cnt * 0.2) + 1]
    validation_non_knowledge_seeking_turns = validation_df[validation_df['target'] == False][
                                             :int(non_knowledge_turn_cnt * 0.2)]
    rede_detector.find_threshold(validation_knowledge_seeking_turns.apply(extract_last_log_text, axis=1).tolist(),
                                 validation_non_knowledge_seeking_turns.apply(extract_last_log_text, axis=1).tolist())

    test_dataset = load_dataset(dataset_repo, split='test')
    test_df = test_dataset.to_pandas()
    test_df = test_df.sample(500, random_state=42)
    test_df['last_question'] = test_df.apply(extract_last_log_text, axis=1)

    test_knowledge_seeking_turns = test_df[test_df['target']]
    test_non_knowledge_seeking_turns = test_df[test_df['target'] == False]

    print('Test Results')
    rede_detector.evaluate(test_knowledge_seeking_turns['last_question'].tolist(),
                           test_non_knowledge_seeking_turns['last_question'].tolist())


if __name__ == '__main__':
    main()
