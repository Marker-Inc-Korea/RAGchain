import click

from KoPrivateGPT.options import Options
from KoPrivateGPT.pipeline import BasicIngestPipeline
from KoPrivateGPT.utils.embed import Embedding


@click.command()
@click.option('--device_type', default='mps', help='device to run on, select gpu, cpu or mps')
@click.option('--vectordb_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='ko-sroberta-multitask',
              help='embedding model to use, select OpenAI or KoSimCSE or ko-sroberta-multitask')
@click.option('--retrieval_type', default='bm25', help='retrieval type to use, select vectordb or bm25')
def main(device_type, vectordb_type, embedding_type, retrieval_type):
    pipeline = BasicIngestPipeline(
        retrieval_type=(retrieval_type, {"save_path": Options.bm25_db_dir,
                                         "vectordb_type": vectordb_type,
                                         "embedding_type": Embedding(embed_type=embedding_type,
                                                                     device_type=device_type),
                                         "device_type": device_type})
    )
    pipeline.run()


if __name__ == "__main__":
    main()
