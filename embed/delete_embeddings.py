import shutil
import pinecone
from vectorDB import DB, DBType
from embedded_files_cache import EmbeddedFilesCache
from options import ChromaOptions, PineconeOptions
import click


@click.command()
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
def main(db_type):
    EmbeddedFilesCache.delete_files()
    db = DB(db_type, embeddings=None).db_type
    if db == DBType.CHROMA:
        shutil.rmtree(ChromaOptions.persist_dir)
    elif db == DBType.PINECONE:
        pinecone.delete_index(PineconeOptions.index_name)
    else:
        raise ValueError(f"Unknown db type: {db_type}")
    print(f"Deleted {db_type} embeddings")


if __name__ == "__main__":
    main()
