import os
import pathlib

from RAGchain.schema import Passage

PASSAGES = [
    Passage(
        id="id-1",
        content="This is about my cat.",
        filepath="filepath-1"
    ),
    Passage(
        id="id-2",
        content="I want to surf on the beach.",
        filepath="filepath-2"
    ),
    Passage(
        id="id-3",
        content="Do you want to go to church?",
        filepath="filepath-3"
    )
]

root_dir = pathlib.PurePath(os.path.dirname(os.path.realpath(__file__))).parent.parent.parent
