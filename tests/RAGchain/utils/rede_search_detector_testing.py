from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.mixture import GaussianMixture

load_dotenv()
from RAGchain.utils.rede_search_detector import RedeSearchDetector


def extract_last_log_text(row):
    return row.log[-1]['text']


def main():
    dataset_repo = "NomaDamas/DSTC-11-Track-5"
    test_dataset = load_dataset(dataset_repo, split='test')
    test_df = test_dataset.to_pandas()
    test_df = test_df.sample(20, random_state=42)

    knowledge_seeking_turns = test_df[test_df['target']]
    non_knowledge_seeking_turns = test_df[test_df['target'] == False]

    rede_detector = RedeSearchDetector(threshold=0.5)
    rede_detector.find_representation_transform([
        row.log[-1]['text'] for i, row in knowledge_seeking_turns.iterrows()
    ])
    rede_detector.train_density_estimation(
        GaussianMixture(n_components=1),
        [row.log[-1]['text'] for i, row in non_knowledge_seeking_turns.iterrows()])
    validation_dataset = load_dataset(dataset_repo, split='validation')
    validation_df = validation_dataset.to_pandas()
    validation_df = validation_df.sample(100, random_state=42)
    validation_knowledge_seeking_turns = validation_df[validation_df['target']]
    validation_non_knowledge_seeking_turns = validation_df[validation_df['target'] == False]
    rede_detector.find_threshold(validation_knowledge_seeking_turns.apply(extract_last_log_text, axis=1).tolist(),
                                 validation_non_knowledge_seeking_turns.apply(extract_last_log_text, axis=1).tolist())


if __name__ == '__main__':
    main()
