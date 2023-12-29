from typing import List

import pandas as pd
from langchain.schema.embeddings import Embeddings
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, KMeans, DBSCAN, MeanShift, OPTICS, \
    SpectralClustering

from RAGchain.schema import Passage
from RAGchain.utils.util import text_modifier


class SemanticClustering:
    """
    This class is used to cluster the passages based on their semantic information.
    First, we vectorize to embedding vector for representing each passages' semantic information.
    Second, we cluster the embedding vectors by using various clustering algorithm.

    There are no optimal clustering algorithm for all cases. So, you can try various clustering algorithm.
    """

    def __init__(self, embedding_function: Embeddings,
                 clustering_algorithm: str = 'kmeans'):
        self.embedding_function = embedding_function
        self.clustering_algorithm = clustering_algorithm

    def cluster(self, passages: List[Passage], **kwargs) -> List[List[Passage]]:
        """
        clustering
        :param passages: list of passages to be clustered.
        :param kwargs: kwargs for clustering algorithm.

        :return: 2-d list of clustered Passages. Each cluster is a list of passages.
        """
        embeddings = self.embedding_function.embed_documents([passage.content for passage in passages])

        clustering_algorithm_dict = {
            'affinity_propagation': AffinityPropagation,
            'agglomerative_clustering': AgglomerativeClustering,
            'birch': Birch,
            'dbscan': DBSCAN,
            'kmeans': KMeans,
            'mean_shift': MeanShift,
            'optics': OPTICS,
            'spectral_clustering': SpectralClustering,
        }

        clustering_algorithm_class = self.__select_clustering_algorithm(clustering_algorithm_dict)
        clustering_algorithm = clustering_algorithm_class(**kwargs)
        clustering_algorithm.fit(embeddings)

        df = pd.DataFrame({
            'id': [passage.id for passage in passages],
            'cluster': clustering_algorithm.labels_.tolist(),
            'passage': passages
        })
        return df.groupby('cluster')['passage'].apply(list).tolist()

    def __select_clustering_algorithm(self, instance_dict: dict):
        algorithm_names = list(instance_dict.keys())
        for modified_name in text_modifier(self.clustering_algorithm):
            if modified_name in algorithm_names:
                return instance_dict[modified_name]
        raise ValueError(f"Clustering algorithm {self.clustering_algorithm} is not supported. "
                         f"Please choose one of {algorithm_names}.")
