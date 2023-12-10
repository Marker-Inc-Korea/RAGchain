from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
from RAGchain.utils.rede_search_detector import RedeSearchDetector


def main():
    dataset_repo = "NomaDamas/DSTC-11-Track-5"
    test_dataset = load_dataset(dataset_repo, split='test')
    test_df = test_dataset.to_pandas()
    test_df = test_df.sample(20, random_state=42)

    knowledge_seeking_turns = test_df[test_df['target']]
    non_knowledge_seeking_turns = test_df[test_df['target'] == False]

    rede_detector = RedeSearchDetector(threshold=0.5)
    rede_detector.representation_transform([
        row.log[-1]['text'] for i, row in knowledge_seeking_turns.iterrows()
    ])


if __name__ == '__main__':
    main()
