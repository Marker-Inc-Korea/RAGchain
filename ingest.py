import click

from KoPrivateGPT.pipeline import BasicIngestPipeline
from KoPrivateGPT.utils.embed import EmbeddingFactory
from config import MongoDBOptions
from config import Options, PickleDBOptions


@click.command()
@click.option('--device_type', default='mps', help='device to run on, select gpu, cpu or mps')
@click.option('--vectordb_type', default='chroma', help='vector database to use, select chroma or pinecone')
@click.option('--embedding_type', default='ko-sroberta-multitask',
              help='embedding model to use, select OpenAI or KoSimCSE or ko-sroberta-multitask')
@click.option('--retrieval_type', default='bm25', help='retrieval type to use, select vectordb or bm25')
@click.option('--db_type', default='mongo_db', help='db type to use, select pickle_db or mongo_db')
def main(device_type, vectordb_type, embedding_type, retrieval_type: str, db_type: str):
    pipeline = BasicIngestPipeline(
        file_loader_type=("file_loader", {
            "target_dir": Options.source_dir,
            "hwp_host_url": Options.HwpConvertHost
        }),
        retrieval_type=(retrieval_type, {
            "save_path": Options.bm25_db_dir,
            "vectordb_type": vectordb_type,
            "embedding_type": EmbeddingFactory(embed_type=embedding_type,
                                               device_type=device_type).get(),
            "device_type": device_type
        }),
        db_type=(db_type, {
            'save_path': PickleDBOptions.save_path,
            "mongo_url": MongoDBOptions.mongo_url,
            "db_name": MongoDBOptions.db_name,
            "collection_name": MongoDBOptions.collection_name
        })
    )
    pipeline.run()


if __name__ == "__main__":
    main()
