import logging

from datasets import load_dataset
from sklearn.mixture import GaussianMixture

from RAGchain.utils.rede_search_detector import RedeSearchDetector

logger = logging.getLogger(__name__)


class SearchDetectorEvaluator:
    """
    You can evaluate search detector performance using this class.
    This class uses DSTC11 Track5 dataset for evaluation. And used metrics for evaluation are precision, recall, f1.
    From now, we only support RedeSearchDetector, because it is only Search Detector that supports in RAGchain.
    You might use more search detector when we add them.
    """

    def __init__(self, search_detector: RedeSearchDetector,
                 random_state: int = 42):
        dataset_repo = "NomaDamas/DSTC-11-Track-5"
        self.train_df = load_dataset(dataset_repo, split='train').to_pandas()
        self.validation_df = load_dataset(dataset_repo, split='validation').to_pandas()
        self.test_df = load_dataset(dataset_repo, split='test').to_pandas()
        self.search_detector = search_detector
        self.random_state = random_state

    def train(self, train_knowledge_seeking_cnt: int = 5,
              train_non_knowledge_seeking_cnt: int = 50,
              valid_percentage: float = 0.2):
        assert valid_percentage <= 1.0, "valid_percentage should be less than 1.0"

        # make train dataset
        train_df = self.train_df.sample((train_knowledge_seeking_cnt + train_non_knowledge_seeking_cnt) * 10,
                                        random_state=self.random_state)
        knowledge_seeking_turns = train_df[train_df['target']][:train_knowledge_seeking_cnt]
        non_knowledge_seeking_turns = train_df[train_df['target'] == False][:train_non_knowledge_seeking_cnt]

        # train
        self.search_detector.find_representation_transform(
            knowledge_seeking_turns.apply(self.extract_last_log_text, axis=1).tolist())
        self.search_detector.train_density_estimation(
            GaussianMixture(n_components=1),
            non_knowledge_seeking_turns.apply(self.extract_last_log_text, axis=1).tolist())

        # make validation dataset
        validation_df = self.validation_df.sample((train_knowledge_seeking_cnt + train_non_knowledge_seeking_cnt) * 2)
        validation_knowledge_seeking_turns = validation_df[self.validation_df['target']][
                                             :int(train_knowledge_seeking_cnt * valid_percentage) + 1]
        validation_non_knowledge_seeking_turns = validation_df[self.validation_df['target'] == False][
                                                 :int(train_non_knowledge_seeking_cnt * valid_percentage)]

        # find threshold using validation dataset + evaluate validation dataset
        logger.info('Validation Results')
        threshold = self.search_detector.find_threshold(
            validation_knowledge_seeking_turns.apply(self.extract_last_log_text, axis=1).tolist(),
            validation_non_knowledge_seeking_turns.apply(self.extract_last_log_text, axis=1).tolist())

        logger.info(f"Threshold is {threshold}")

    def evaluate(self, test_cnt: int = 500):
        # make test dataset
        test_df = self.test_df.sample(test_cnt, random_state=self.random_state)
        test_knowledge_seeking_turns = test_df[test_df['target']]
        test_non_knowledge_seeking_turns = test_df[test_df['target'] == False]

        # evaluate test dataset
        logger.info('Test Results')
        precision, recall, f1 = self.search_detector.evaluate(
            test_knowledge_seeking_turns.apply(self.extract_last_log_text, axis=1).tolist(),
            test_non_knowledge_seeking_turns.apply(self.extract_last_log_text, axis=1).tolist())
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"F1: {f1}")
        return precision, recall, f1

    def extract_last_log_text(self, row):
        return row.log[-1]['text']
