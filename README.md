# KoPrivateGPT

English Version of README is [here](./docs/README_en.md). Please follow the link to read the English version of README.

본 프로젝트는 [privateGPT](https://github.com/imartinez/privateGPT)와 [localGPT](https://github.com/PromtEngineer/localGPT)에서 영감을 받아 만들어졌습니다. 

해당 프로젝트들은 여러 문서들을 벡터 DB에 저장을 하여, 문서 내용에 대해 LLM과 대화할 수 있는 프로젝트입니다. 오픈소스 Document Q&A라고 보시면 됩니다.

그러나 해당 프로젝트들은 한국에서 활용하기 어려울 정도의 성능을 한국어에서 보여줍니다. 이를 해결하기 위하여 한국어 버전의 KoPrivateGPT 프로젝트를 만들었습니다.

한국어를 이용하여 오프라인에서도 모든 문서들에 대해 LLM에게 질문을 해보세요. 100% 프라이버시가 보장되며, 어떠한 데이터도 외부로 전송되지 않습니다. 인터넷 연결 없이 문서를 불러오고 질문을 해보세요!

(더 좋은 성능 확보를 위해 OpenAI의 GPT 모델을 사용할 수도 있습니다. 다만, 이 경우 인터넷 연결이 필요하며 프라이버시가 보장되지 않습니다.)

원래 프로젝트에서 한국에 맞게 변형한 부분은 현재까지 다음과 같습니다. 
- 한국어 모델 [KoAlpaca](https://github.com/Beomi/KoAlpaca) 적용
- 한국어 모델 [KuLLM](https://github.com/nlpai-lab/KULLM) 적용 (동작하나, 성능이 좋지 않아 KoAlpaca 사용을 추천합니다)
- 한국어 임베딩 [Korean-Sentence-Embedding](https://github.com/BM-K/Sentence-Embedding-Is-All-You-Need) 적용
- HWP 파일 문서 호환 추가 ([hwp-converter-api](https://github.com/edai-club/hwp-converter-api) 사용)

[//]: # (## Colab 데모)

[//]: # (콜랩에서 실행할 수 있는 데모 버전을 준비하였습니다. 아쉽게도 콜랩 버전에서 HWP 파일은 사용할 수 없습니다. )

[//]: # ([여기]&#40;https://colab.research.google.com/drive/1wFV8WSfna0p1HYD_N8KmlrB69ItWczsZ?usp=sharing&#41;에서 콜랩 데모 버전을 실행해보세요.)

[//]: # (<a style='display:inline' target="_blank" href="https://colab.research.google.com/drive/1wFV8WSfna0p1HYD_N8KmlrB69ItWczsZ?usp=sharing">)

[//]: # (  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>)

[//]: # (</a>)

[//]: TODO : Edit Colab Demo for latest version.

## Docker를 통한 실행법 (추천)
아래 코드를 실행하여 도커 컨테이너를 통해 KoPrivateGPT를 실행할 수 있습니다.
```shell
git clone https://github.com/edai-club/KoPrivateGPT.git
cd KoPrivateGPT
docker compose up
```
이후 `http://localhost:7860`으로 접속하여 KoPrivateGPT를 사용할 수 있습니다.

### 주의
*docker compose up*을 실행하기 전에 반드시 .env.template 파일을 참고하여 .env 파일을 만들어 주세요. 
또한, Docker를 통해 실행되는 Web UI는 현재 Pinecone 및 OpenAI 버전만 지원합니다. 

# 직접 빌드하여 실행하기
해당 프로젝트를 직접 실행하기 위해서 아래의 코드를 실행해주세요. 파이썬 3.10 이상이 설치된 환경에서의 실행을 추천합니다.

```shell
git clone https://github.com/edai-club/KoPrivateGPT.git
cd KoPrivateGPT
pip install -r requirements.txt
```

## 한글파일 사용법
  KoPrivateGPT 또한 도커 컨테이너로 사용하는 상황을 바탕으로 만들어졌습니다.
### Docker
도커를 다운 받고 hwp converter api 서버용 로컬 컨테이너를 만들어줍니다.
- [Docker Hub](https://hub.docker.com/r/vkehfdl1/hwp-converter-api)
```bash
docker run -it -d --name hwp-converter -p (외부포트):7000 vkehfdl1/hwp-converter-api:1.0.0
```
### 터미널
터미널에서 KoPrivateGPT 컨테이너와 hwp-converter 컨테이너를 연결합니다.
```bash
#docker network create <NETWORK_NAME>
docker network create docs-convert-api-net
#docker network connect <NETWORK_NAME> <CONTAINER_NAME>
docker network connect docs-convert-api-net KoPrivateGPT
docker network connect docs-convert-api-net hwp-converter
```
### ingest.py
`KoPrivateGPT/ingest.py`로 이동하여 http request 주소를 확인합니다.
```python
# ingest.py line 13-14
HwpConvertOpt = 'all'# 'main-only'
HwpConvertHost = f'http://hwp-converter:7000/upload?option={HwpConvertOpt}'
```
실행중인 도커 컨테이너의 이름과 내부포트를 맞춰 설정해줍니다.
`HwpConvertOpt`를 `all` 로 설정하면 hwp파일 속 '표'의 데이터를 포함한 텍스트 데이터를 반환합니다.
`main-only`는 hwp파일 속 '표'를 제외한 텍스트 데이터를 반환합니다.
이후 다른 파일들과 같이 `SOURCE_DOCUMENTS` 폴더 안에 hwp 파일을 넣으면
`ingest.py`가 동작할 때 hwp 파일을 텍스트로 처리할 수 있습니다.

### 주의
- hwpx 파일은 지원하지 않습니다. hwp 파일로 변환하여 시도해주세요.

# 로컬 모델 사용법

KoPrivateGPT는 OpenAI API 기반으로 제작되었으며, OpenAI API-ish한 모든 로컬 모델들을 사용할 수 있습니다.
사용을 위해서는 api_base에 원하는 url로 바꾸어서 넣어주세요.
로컬 모델을 사용하는 방법에 따른 가이드는 아래와 같습니다.

- [vLLM 가이드](./docs/vLLM_guide.md)

그 외에도 [LocalAI](https://localai.io/basics/getting_started/), [LiteLLM](https://github.com/BerriAI/litellm) 서버 역시 사용할 수
있습니다.


# 직접 원하는 문서를 불러오는 법
SOURCE_DOCUMENTS 폴더 안에 원하는 .txt, .pdf, .csv, .hwp, 혹은 .xlsx 파일을 넣어주세요.

현재는 .txt, .pdf, .csv, .xlsx, .hwp 파일만 지원합니다. 다른 파일을 사용하고 싶으시면, 해당 파일을 지원하는 파일로 변환을 해야 합니다.

그리고 아래 코드를 이용하여 모든 문서를 불러옵니다. 
    
 ```shell  
python ingest.py
 ```
위 코드를 실행하면 문서들을 텍스트 뭉치로 자른 뒤, 한국어 임베딩을 사용하여 벡터로 임베딩하고 그것을 chroma 벡터 DB로 저장합니다. 
저장은 DB 폴더에 되며, 만약 모든 데이터를 삭제하고 싶다면 DB 폴더를 완전히 삭제하면 됩니다.

# LLM에게 질문하는 법
질문을 하기 위해서는 다음과 같이 실행하면 됩니다.

```shell
python run_localGPT.py
```
이후 아래 명령어가 나오면, 원하는 질문을 입력하면 됩니다. 
```shell
> 질문:
```

위 코드를 실행하면 문서들을 텍스트 뭉치로 자른 뒤, 한국어 임베딩을 사용하여 벡터로 임베딩하고 그것을 chroma 벡터 DB로 저장합니다. 저장은 DB 폴더에 되며, 만약 모든 데이터를 삭제하고 싶다면 DB 폴더를 완전히 삭제하면 됩니다. '
조금 기다리면 인공지능의 답변이 제공되며, 해당 답변을 제공하는데에 참조한 문서 4개의 출처와 그 내용이 출력됩니다. 

'exit' 혹은 '종료' 를 입력하면 실행이 종료됩니다.

### KuLLM 모델 사용법
KuLLM 구동을 위해서는 아래 코드를 입력하세요.
```shell
python run_localGPT.py --model_type=KuLLM
```

현재 동작하지만, 성능이 좋지 않기 때문에 로컬 모델 사용시에서는 KoAlpaca 사용을 추천합니다. 

## OpenAI 모델 사용법
기기의 성능이 부족해 KoAlpaca 구동에 실패하였다면, OpenAI 모델을 사용하세요.
데이터가 OpenAI에 제공되어 완전히 private 하지는 않지만, 낮은 성능의 기기에서도 실행할 수 있습니다. 

아래의 코드에 본인의 OpenAI 토큰을 넣어 구동할 수 있습니다.
```shell
python run_localGPT.py --model_type=openai --openai-token=<Your OPENAI TOKEN>
```

### CPU에서 구동 방법
vram 부족 등의 이유로 cuda 기반의 GPU가 아닌 CPU에서 모델을 돌리고자 한다면, 아래처럼 --device_type 옵션을 사용하세요.
```shell
python run_localGPT.py --device_type=cpu --model_type=<model_name>
```

ggml을 이용해 양자화한 5bit 모델을 Ctransformers를 이용해 작동시킵니다.

# 질문 및 답변 예시 - KoAlpaca Polyglot
KoAlpaca 모델로 cpu에서 실행한 결과입니다. 
```markdown
> 질문 : 
제주2공항이 어디에 있어?

> 대답 : 
제주2공항은 제주도 서귀포시 성산읍에 위치해 있습니다.
```


```markdown
> 질문 : 
제주2공항의 설치 목적이 무엇이지?

> 대답 : 
제주2공항은 현재 군용기와 여객기가 모두 이용하는 공군비행장으로 쓰고 있는 공간을 민간용으로 활용하기 위해 설치됩니다. 해당 공항의 건설 목적은 항공운송 경쟁력 강화 및 지역경제 발전입니다.
```

```markdown
> 질문 :
제주2공항의 규모는 얼마나 되나요?

> 대답 :
제주 제 2 공항은 현재 공항과 비교했을 때 부지 면적이 약 3배 이상 (약 580만 평방미터)에 달하며, 활주로 1분, 여객터미널 및 화물터미널 등 다양한 시설이 건설될 예정입니다. 또한, 운영 계획에 따라
유동적이지만, 연간 약 2500만 명 (국내선 1000만 명, 국제선 500만 명)의 처리 능력을 가질 것으로 예상됩니다. 참고로, 현재 제주공항은 부지 면적이 약 3배 이상 늘어났으며, 활주로 2개 (각각
800m)가 추가 건설되었습니다.
```

현재 계속해서 성능 개선 중에 있으며, 더욱 좋은 성능 확보를 위해서는 OpenAI의 GPT 모델 사용을 추천합니다.

# 평가

evaluate/strategyQA 폴더를 참고해 주세요.

# 기타 사용법

## 임베딩한 벡터 삭제법

아래와 같이 실행하면 DB에 저장된 임베딩된 벡터와, 임베딩 기록 파일을 모두 삭제할 수 있습니다.

```shell
python3 delete_embeddings.py --db_type=<DB name : pinecone or chroma>
```

# 시스템 요구 사항

## 파이썬 버전

이 소프트웨어를 사용하려면 Python 3.10 이상이 설치되어 있어야 합니다. 이전 버전의 Python은 컴파일되지 않을 수 있습니다.

## C++ 컴파일러

'pip install'을 하는 중에 오류가 발생하면 컴퓨터에 C++ 컴파일러를 설치해야 할 수 있습니다.

### Windows 10/11의 경우
Windows 10/11에서 C++ 컴파일러를 설치하려면 다음 단계를 따르세요:

1. Visual Studio 2022를 설치합니다.
2. 다음 컴포넌트가 선택되어 있는지 확인합니다:
   * 범용 Windows 플랫폼 개발
   * Windows용 C++ CMake 툴
3. MinGW 홈페이지](https://sourceforge.net/projects/mingw/)에서 MinGW 설치 파일을 다운로드합니다.
4. 설치 파일을 실행하고 "gcc" 컴포넌트를 선택합니다.

### NVIDIA 드라이버 문제:
이 [페이지](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-22-04)를 참고하여 NVIDIA 드라이버를 설치합니다.
        

# 면책 조항
이 프로젝트는 LLM 및 벡터 임베딩을 사용하여 한국어로 질문에 답변할 수 있는 완전한 로컬 솔루션의 가능성을 검증하기 위한 시험용 프로젝트입니다. 
프로덕션 활용을 위한 준비는 완료되지 않았으며 프로덕션에 사용할 수 없습니다.
