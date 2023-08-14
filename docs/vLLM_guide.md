# vLLM openAI API 서버 실행하기

vLLM은 기존 방법에 비해 최대 24배 빠른 로컬 모델의 인퍼런스 속도를 가진 라이브러리입니다. vLLM을 이용하여 KoPrivateGPT를 활용하세요.

## vLLM 설치하기

공식 방법은 [vLLM 홈페이지]에서 확인하세요.

```bash
pip install fschat accelerate vllm
```

필요한 라이브러리를 설치합니다. 이후, 아래 코드를 통해 openai API 서버를 실행합니다.

```bash
python -m vllm.entrypoints.openai.api_server --model <model_name> --port <port_number> --host <host_name>
```

로컬에서 실행한다면 host_name은 0.0.0.0으로 설정하세요. port_number는 기본적으로 8000으로 설정되어 있습니다.

model_name은 huggingface에 올라와 있는 모델을 활용하거나, 로컬 모델의 경로를 입력할 수 있습니다.

한국어 모델의 model_name은 아래와 같습니다.

- KoAlpaca : beomi/KoAlpaca-Polyglot-12.8B
- KuLLM : nlpai-lab/kullm-polyglot-12.8b-v2
- Polyglot-ko : EleutherAI/polyglot-ko-12.8b

## Colab에서 실행

<a style='display:inline' target="_blank" href="https://colab.research.google.com/drive/1ICSL0lVCavh2TyW4mbeI7uVka5FGYJRx?usp=sharing">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

[Colab](https://colab.research.google.com/drive/1ICSL0lVCavh2TyW4mbeI7uVka5FGYJRx?usp=sharing)에서 vLLM 서버를 실행할 수 있습니다.

## KoPrivateGPT에서 사용하기

run_localGPT.py를 실행할 때 --api_base 옵션에 vLLM API 서버의 주소를 입력하면 됩니다.  
