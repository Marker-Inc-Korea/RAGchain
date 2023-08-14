# This guide is under construction!

# 이 가이드는 작업중입니다.

# 우바부가 openAI API 서버 실행하기

우바부가는 웹 UI와 로컬 모델 실행을 쉽게 할 수 있는 라이브러리입니다.
우바부가의 openai extension을 활용하면 KoPrivateGPT에서도 우바부가의 로컬 모델을 바로 활용할 수 있습니다.

## 실행법

우바부가 openai extension의 공식 실행법은 [여기서](https://github.com/oobabooga/text-generation-webui/tree/main/extensions/openai)
확인하세요.
또한, 우바부가 설치 방법은 [여기서](https://github.com/oobabooga/text-generation-webui) 확인하세요.

## 한국어 모델 작동 예시 (ggml)

ggml은 CUDA 없이도 로컬 모델을 사용할 수 있게 양자화 해주는 라이브러리이며, ggml을 통해 양자화 한 모델은 CTransformers 등을 이용해 실행할 수 있습니다.
한국어 오픈소스 모델인 Polyglot-ko는 GPT-NeoX 아키텍쳐이며, 이는 CTransformers에서 인퍼런스를 지원합니다.
따라서, ggml을 통해 양자화 한 Polyglot-ko 기반 한국어 모델을 CTransformers를 이용해 실행할 수 있습니다.

아래는 CTransformers를 이용하여 KoAlpaca를 실행하는 예시입니다.

```bash
python3 download-model.py vkehfdl1/KoAlpaca-Polyglot-12.8b-ggml
python3 server.py --model vkehfdl1/KoAlpaca-Polyglot-12.8b-ggml --loader ctransformers \
                  --model_type gpt_neox --listen --auto-devices --api --extensions openai \ 
                  --share
```

## KoPrivateGPT에서 사용하기

run_localGPT.py를 실행할 때 --api_base 옵션에 vLLM API 서버의 주소를 입력하면 됩니다.
