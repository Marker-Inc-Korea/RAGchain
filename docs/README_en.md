# KoPrivateGPT

This project was inspired by the original [privateGPT](https://github.com/imartinez/privateGPT)
and [localGPT](https://github.com/PromtEngineer/localGPT). Most of the description here is inspired by the original
privateGPT and original localGPT.

In this model, I have replaced the GPT4ALL model with KoAlpaca-Polyglot model, and we are using the Korean Sentence
Embeddings instead of LlamaEmbeddings as used in the original privateGPT.

Plus, we add HWP converter for ingesting HWP files that crucial to Korean businessmen and women.

Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves
your execution environment at any point. You can ingest documents and ask questions without an internet connection!
(For better performance, you can also use OpenAI's GPT model. However, this requires an internet connection and does not
guarantee privacy.)

Built with [LangChain](https://github.com/hwchase17/langchain) and [KoAlpaca](https://github.com/Beomi/KoAlpaca)
and [Korean-Sentence-Embedding](https://github.com/BM-K/Sentence-Embedding-Is-All-You-Need)

# Environment Setup

In order to set your environment up to run the code here, first install all requirements:

```shell
git clone https://github.com/edai-club/KoPrivateGPT.git
cd RAGchain
pip install -r requirements.txt
```

## Test dataset

This repo uses a [제주 제2항 기본계획(안) 보도자료](https://www.korea.kr/common/download.do?fileId=197236015&tblKey=GMN) as an
example.

## Instructions for ingesting your own dataset

Put any and all of your .txt, .pdf, .csv or .hwp files into the SOURCE_DOCUMENTS directory
in the load_documents() function, replace the docs_path with the absolute path of your source_documents directory.

The current default file types are .txt, .pdf, .csv, .xlsx, .hwp, if you want to use any other file type, you will need
to convert it to one of the default file types.

Run the following command to ingest all the data.

```shell
python ingest.py
```

It will create an index containing the local vectorstore. Will take time, depending on the size of your documents.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database.
If you want to start from an empty database, delete the `index`.

Note: When you run this for the first time, it will download take time as it has to download the embedding model. In the
subseqeunt runs, no data will leave your local enviroment and can be run without internet connection.

## Ask questions to your documents, locally!

In order to ask a question, run a command like:

```shell
python run_localGPT.py
```

And wait for the script to require your input.

```shell
> 질문:
```

Hit enter. Wait while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and
the 4 sources it used as context from your documents; you can then ask another question without re-running the script,
just wait for the prompt again.

Note: When you run this for the first time, it will need internet connection to download the KoAlpaca model. After that
you can turn off your internet connection, and the script inference would still work. No data gets out of your local
environment.

Type `exit` to finish the script.

## Use OpenAI Model

If your device is not capable of run KoAlpaca, use OpenAI model.
Using OpenAI Model is not private, but it can run any device with low performance and have better performance.

Type your OpenAI API Token in the below code.

```shell
python run_localGPT.py --model_type=openai --openai-token=<Your OPENAI TOKEN>
```

# System Requirements

## Python Version

To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++
compiler on your computer.

### For Windows 10/11

To install a C++ compiler on Windows 10/11, follow these steps:

1. Install Visual Studio 2022.
2. Make sure the following components are selected:
    * Universal Windows Platform development
    * C++ CMake tools for Windows
3. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/).
4. Run the installer and select the "gcc" component.

### NVIDIA Driver's Issues:

Follow this [page](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04) to install NVIDIA Drivers.

# Disclaimer

This is a test project to validate the feasibility of a fully local solution for question answering using LLMs and
Vector embeddings for Korean. It is not production ready, and it is not meant to be used in production. Vicuna-7B is
based on the Llama model so that has the original Llama license. 
