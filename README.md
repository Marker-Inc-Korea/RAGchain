# 중요!
해당 레포는 아직 개발 중입니다!

# KoPrivateGPT

본 프로젝트는 [privateGPT](https://github.com/imartinez/privateGPT)와 [localGPT](https://github.com/PromtEngineer/localGPT)에서 영감을 받아 만들어졌습니다. 

해당 프로젝트들은 여러 문서들을 벡터 DB에 저장을 하여, 문서 내용에 대해 LLM과 대화할 수 있는 프로젝트입니다. 오픈소스 Document Q&A라고 보시면 됩니다.

그러나 해당 프로젝트들은 한국에서 활용하기 어려울 정도의 성능을 한국어에서 보여줍니다. 이를 해결하기 위하여 한국어 버전의 KoPrivateGPT 프로젝트를 만들었습니다.

한국어를 이용하여 오프라인에서도 모든 문서들에 대해 LLM에게 질문을 해보세요. 100% 프라이버시가 보장되며, 어떠한 데이터도 외부로 전송되지 않습니다. 인터넷 연결 없이 문서를 불러오고 질문을 해보세요!

원래 프로젝트에서 한국에 맞게 변형한 부분은 현재까지 다음과 같습니다. 
- 한국어 모델 [KoAlpaca](https://github.com/Beomi/KoAlpaca) 적용
- 한국어 임베딩 [Korean-Sentence-Embedding](https://github.com/BM-K/Sentence-Embedding-Is-All-You-Need) 적용
- HWP 파일 문서 호환 추가 ([hwp-converter-api](https://github.com/edai-club/hwp-converter-api) 사용)

# 환경 설정
해당 프로젝트를 실행하기 위해서는 다음과 같이 환경 설정을 해주셔야 합니다.

```shell
pip install -r requirements.txt
```

## 테스트 데이터
해당 레포에서는 [대한민국 상법](https://constitutioncenter.org/media/files/constitution.pdf)을 예시로 사용합니다.

# 직접 원하는 문서를 불러오는 법
SOURCE_DOCUMENTS 폴더 안에 원하는 .txt, .pdf, .csv, .hwp, 혹은 .xlsx 파일을 넣어주세요.

현재는 .txt, .pdf, .csv, .xlsx, .hwp 파일만 지원합니다. 다른 파일을 사용하고 싶으시면, 해당 파일을 지원하는 파일로 변환을 해야 합니다.

그리고 아래 코드를 이용하여 모든 문서를 불러옵니다. 
    
 ```shell  
    python ingest.py
 ```

# LLM에게 질문하는 법
질문을 하기 위해서는 다음과 같이 실행하면 됩니다.

```shell
python run_localGPT.py
```
이후 아래 명령어가 나오면, 원하는 질문을 입력하면 됩니다. 
```shell
> Enter a query:
```


# KoPrivateGPT

This project was inspired by the original [privateGPT](https://github.com/imartinez/privateGPT) and [localGPT](https://github.com/PromtEngineer/localGPT). Most of the description here is inspired by the original privateGPT and original localGPT.

In this model, I have replaced the GPT4ALL model with KoAlpaca-Polyglot model, and we are using the Korean Sentence Embeddings instead of LlamaEmbeddings as used in the original privateGPT.

Plus, we add HWP converter for ingesting HWP files that crucial to Korean businessmen and women.

Ask questions to your documents without an internet connection, using the power of LLMs. 100% private, no data leaves your execution environment at any point. You can ingest documents and ask questions without an internet connection!

Built with [LangChain](https://github.com/hwchase17/langchain) and [KoAlpaca](https://github.com/Beomi/KoAlpaca) and [Korean-Sentence-Embedding](https://github.com/BM-K/Sentence-Embedding-Is-All-You-Need)


# Environment Setup
In order to set your environment up to run the code here, first install all requirements:

```shell
pip install -r requirements.txt
```

## Test dataset
This repo uses a [대한민국 상법](https://constitutioncenter.org/media/files/constitution.pdf) as an example.

## Instructions for ingesting your own dataset

Put any and all of your .txt, .pdf, .csv or .hwp files into the SOURCE_DOCUMENTS directory
in the load_documents() function, replace the docs_path with the absolute path of your source_documents directory. 

The current default file types are .txt, .pdf, .csv, and .xlsx, if you want to use any other file type, you will need to convert it to one of the default file types.


Run the following command to ingest all the data.

```shell
python ingest.py
```

It will create an index containing the local vectorstore. Will take time, depending on the size of your documents.
You can ingest as many documents as you want, and all will be accumulated in the local embeddings database. 
If you want to start from an empty database, delete the `index`.

Note: When you run this for the first time, it will download take time as it has to download the embedding model. In the subseqeunt runs, no data will leave your local enviroment and can be run without internet connection.



## Ask questions to your documents, locally!
In order to ask a question, run a command like:

```shell
python run_localGPT.py
```

And wait for the script to require your input. 

```shell
> Enter a query:
```

Hit enter. Wait while the LLM model consumes the prompt and prepares the answer. Once done, it will print the answer and the 4 sources it used as context from your documents; you can then ask another question without re-running the script, just wait for the prompt again. 

Note: When you run this for the first time, it will need internet connection to download the KoAlpaca model. After that you can turn off your internet connection, and the script inference would still work. No data gets out of your local environment.

Type `exit` to finish the script.

# Run it on CPU
By default, localGPT will use your GPU to run both the `ingest.py` and `run_localGPT.py` scripts. But if you do not have a GPU and want to run this on CPU, now you can do that (Warning: Its going to be slow!). You will need to use `--device_type cpu`flag with both scripts. 

For Ingestion run the following: 
```shell
python ingest.py --device_type cpu
```
In order to ask a question, run a command like:

```shell
python run_localGPT.py --device_type cpu
```

# How does it work?
Selecting the right local models and the power of `LangChain` you can run the entire pipeline locally, without any data leaving your environment, and with reasonable performance.

- `ingest.py` uses `LangChain` tools to parse the document and create embeddings locally using `InstructorEmbeddings`. It then stores the result in a local vector database using `Chroma` vector store. 
- `run_localGPT.py` uses a local LLM (Vicuna-7B in this case) to understand questions and create answers. The context for the answers is extracted from the local vector store using a similarity search to locate the right piece of context from the docs.
- You can replace this local LLM with any other LLM from the HuggingFace. Make sure whatever LLM you select is in the HF format.

# System Requirements

## Python Version
To use this software, you must have Python 3.10 or later installed. Earlier versions of Python will not compile.

## C++ Compiler
If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

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
This is a test project to validate the feasibility of a fully local solution for question answering using LLMs and Vector embeddings. It is not production ready, and it is not meant to be used in production. Vicuna-7B is based on the Llama model so that has the original Llama license. 
