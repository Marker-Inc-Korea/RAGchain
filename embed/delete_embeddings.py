import shutil
import pinecone
from vectorDB import DB, DBType
from embed import EmbeddedFilesCache
from options import ChromaOptions, PineconeOptions
import click


@click.command()
@click.option('--db_type', default='chroma', help='vector database to use, select chroma or pinecone')
def main(db_type):
    db = DB(db_type, embeddings=None).db_type
    delete_embeddings(db)


def delete_embeddings(db_type):
    EmbeddedFilesCache.delete_files()
    if db_type == DBType.CHROMA:
        shutil.rmtree(str(ChromaOptions.persist_dir))
    elif db_type == DBType.PINECONE:
        pinecone.delete_index(PineconeOptions.index_name)
    else:
        raise ValueError(f"Unknown db type: {db_type}")
    print(f"Deleted {db_type} embeddings")


if __name__ == "__main__":
    main()
