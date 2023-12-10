from typing import List, Optional

import numpy as np
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema.embeddings import Embeddings


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

    def representation_transform(self,
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

        # TODO: add L truncation
        # find omega matrix
        self.omega_matrix = np.dot(U, np.sqrt(np.linalg.inv(np.diag(S))))

        print("REDE representation transform done.")
