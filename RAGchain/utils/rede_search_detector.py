from typing import List, Optional

import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.mixture import GaussianMixture


def _normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class RedeSearchDetector:
    """
    This class is implementation of REDE, the method for detect knowledge-seeking turn in few-shot setting.
    It contains train function for your custom model, and inference function for detect knowledge-seeking turn.
    You will need non-knowledge seeking turn dialogues. Plus, it will be great you have few knowledge-seeking turn dialogues.

    The method is implementation of below paper:

    @article{jin2021towards,
      title={Towards zero and few-shot knowledge-seeking turn detection in task-orientated dialogue systems},
      author={Jin, Di and Gao, Shuyang and Kim, Seokhwan and Liu, Yang and Hakkani-Tur, Dilek},
      journal={arXiv preprint arXiv:2109.08820},
      year={2021}
    }
    """

    def __init__(self,
                 threshold: float,
                 embedding: Embeddings = OpenAIEmbeddings()):
        """
        :param embedding: Encoder model for encoding sentences to vectors. Langchain Embeddings class. Default is OpenAIEmbeddings.
        :param threshold: Threshold for classify knowledge-seeking turn. If the score is higher than threshold, classify as non-knowledge-seeking turn.
        Find this threshold by using training data that you own. (e.g. 0.5)
        """
        self.embedding = embedding  # Encoder model for encoding sentences to vectors
        self.threshold = threshold
        self.mu = None
        self.omega_matrix = None  # Omega matrix for linear transformation.
        self.gmm = None  # Gaussian Mixture Model for classify knowledge-seeking turn.
        self.norm = None  # Norm for normalize to unit vector.

    def find_representation_transform(self,
                                      knowledge_seeking_sentences: List[str],
                                      L: Optional[int] = None,
                                      ):
        """
        :param knowledge_seeking_sentences: Knowledge-seeking turn sentences. List[str].
        :param L: Number of dimensions of the transformed representation. If None, use whole dimension.
        Default is None.
        """
        # find mu
        vectors = np.array(self.embedding.embed_documents(knowledge_seeking_sentences))
        self.mu = np.mean(vectors, axis=0)

        # get covariance matrix
        sigma = np.cov(vectors.T)

        # singular value decomposition
        U, S, V = np.linalg.svd(sigma)

        # find omega matrix
        self.omega_matrix = U @ np.sqrt(np.linalg.inv(np.diag(S)))
        if L is not None:
            self.omega_matrix = self.omega_matrix[:, :L]

        print("REDE representation transform done.")

    def representation_formation(self, vectors: np.ndarray) -> np.ndarray:
        """
        :param vectors: Vectors after encoding. np.ndarray.
        :return: Transformed vectors. np.ndarray.
        """
        return (vectors - self.mu) @ self.omega_matrix

    def train_density_estimation(self,
                                 gmm: GaussianMixture,
                                 non_knowledge_seeking_sentences: List[str]):
        """
        :param gmm: Gaussian Mixture Model for classify knowledge-seeking turn. GaussianMixture. n_components must be 1.
        :param non_knowledge_seeking_sentences: Non-knowledge-seeking turn sentences. List[str].
        """
        self.gmm = gmm
        sentence_vectors = np.array(self.embedding.embed_documents(non_knowledge_seeking_sentences))
        transformed_vectors = np.array(
            [self.representation_formation(sentence_vector) for sentence_vector in sentence_vectors])
        # normalize to unit vector
        transformed_vectors = _normalize_vectors(transformed_vectors)

        self.gmm.fit(transformed_vectors)

    def find_threshold(self,
                       valid_knowledge_seeking_sentences: List[str],
                       valid_non_knowledge_seeking_sentences: List[str]):
        """
        Find threshold using Youden's index from validation data predictions.
        :param valid_knowledge_seeking_sentences: knowledge-seeking turn sentences for validation. List[str].
        You can put same sentences that you used for find_representation_transform function.
        :param valid_non_knowledge_seeking_sentences: non-knowledge-seeking turn sentences for validation. List[str].
        """
        true_scores = self._get_density_score(valid_knowledge_seeking_sentences)
        false_scores = self._get_density_score(valid_non_knowledge_seeking_sentences)

        y_true = np.concatenate([np.ones_like(true_scores), np.zeros_like(false_scores)])
        y_score = true_scores + false_scores

        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        idx = np.argmax(fpr - tpr)
        self.threshold = thresholds[idx]

        precision, recall, f1 = self._calculate_metrics(y_true, y_score)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")

        return self.threshold

    def detect(self, sentences: List[str]) -> bool:
        """
        :param sentences: Sentences to detect. List[str].
        :return: True if the sentence is knowledge-seeking turn, else False. bool.
        """
        score = self._get_density_score(sentences)[0]
        return score < self.threshold

    def evaluate(self, test_knowledge_seeking_sentences: List[str],
                 test_non_knowledge_seeking_sentences: List[str]):
        """
        Evaluate rede search detector using test dataset.
        :param test_knowledge_seeking_sentences: knowledge-seeking turn sentences for test. List[str].
        :param test_non_knowledge_seeking_sentences: non-knowledge-seeking turn sentences for test. List[str].
        """
        true_scores = self._get_density_score(test_knowledge_seeking_sentences)
        false_scores = self._get_density_score(test_non_knowledge_seeking_sentences)

        y_true = np.concatenate([np.ones_like(true_scores), np.zeros_like(false_scores)])
        y_score = true_scores + false_scores

        precision, recall, f1 = self._calculate_metrics(y_true, y_score)
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")

        return precision, recall, f1

    def _get_density_score(self, sentences: List[str]) -> List[float]:
        sentence_vectors = np.array(self.embedding.embed_documents(sentences))
        transformed_vectors = np.array([self.representation_formation(np.array(v)) for v in sentence_vectors])
        transformed_vectors = _normalize_vectors(transformed_vectors)
        scores = self._score_vectors(transformed_vectors)
        return scores

    def _score_vectors(self, vectors):
        return [self.gmm.score(vector.reshape(1, -1)) for vector in vectors]

    def _calculate_metrics(self, y_true, y_score):
        predictions = np.where(y_score < self.threshold, 1, 0)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        f1 = f1_score(y_true, predictions)
        return precision, recall, f1
