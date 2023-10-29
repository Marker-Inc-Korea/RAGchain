from glob import glob
from os.path import basename, splitext

from setuptools import find_packages, setup

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='RAGchain',
    version='0.1.3',
    description='Build advanced RAG workflows with LLM, compatible with Langchain',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='NomaDamas',
    author_email='vkehfdl1@gmail.com',
    keywords=['RAG', 'RAGchain', 'ragchain', 'LLM', 'Langchain', 'DQA', 'GPT', 'ODQA'],
    python_requires='>=3.8',
    packages=find_packages(where='.'),
    package_dir={'': '.'},
    url="https://github.com/NomaDamas/RAGchain",
    license='Apache License 2.0',
    py_modules=[splitext(basename(path))[0] for path in glob('./*.py')],
    install_requires=[
        'langchain==0.0.324',
        'chromadb>=0.4.15',
        'urllib3',
        'pdfminer.six',
        'click',
        'openpyxl',
        'pinecone-client',
        'python-dotenv==1.0.0',
        'tiktoken',
        'rank_bm25',
        'numpy',
        'pandas',
        'pydantic',
        'tqdm',
        'pymongo',
        'requests',
        'redis',
        'aiohttp',
        'openai',
        'InstructorEmbedding',
        'sentence-transformers',
        'huggingface_hub',
        'transformers',
        'torch',
        'pyarrow',
        'fastparquet',
        'ragas',
        'datasets',
        'lxml',
        'tiktoken',
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ]
)
