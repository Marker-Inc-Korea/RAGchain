from typing import List

from KoPrivateGPT.schema import Vector

TEST_VECTORS: List[Vector] = [
    Vector(
        vector=[0.5, 0.6, 0.7],
        passage_id="test1"
    ),
    Vector(
        vector=[0.1, 0.2, 0.3],
        passage_id="test2"
    ),
    Vector(
        vector=[0.4, 0.7, 0.9],
        passage_id="test3"
    )
]
