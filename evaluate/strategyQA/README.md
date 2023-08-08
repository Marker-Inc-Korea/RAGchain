# Evaluate with Ko-StrategyQA
Ko-StrategyQA를 이용하여 retreive 성능 및 LLM의 성능을 평가합니다.
Ko-StrategyQA에 대한 자세한 설명은 [허깅페이스](https://huggingface.co/datasets/vkehfdl1/Ko-StrategyQA)를 참고해주세요.

## 평가 방법
평가 전, prediction 파일을 json 형태로 만들어 주세요. predictions_small.json 파일은 prediction 파일의 예시입니다. 해당 파일과 같은 형태로 만들어주면 됩니다. 


### prediction 파일 설명
```json
{
    "e0044a7b4d146d611e73": {
        "answer": false,
        "decomposition": [
            "What is the population of Albany, Georgia?",
            "How many people are in the Albany, New York?  Is #1 greater than or equal to #2?"
        ],
        "paragraphs": [
            "Albany, New York-7",
            "North Albany, Albany, New York-10",
            "Albany, New York-13",
            "Albany, New York-24",
            "Albany, New York-19",
            "Albany, New York-15",
            "Albany, Georgia-34",
            "Albany, Georgia-36",
            "Albany, Georgia-35",
            "Albany, Georgia-40"
        ]
    },
>>>>>>> 이하 생략 <<<<<<<<
}
```
- 위와 같이 answer에는 True 혹은 false로 LLM이 예측한 정답을 정제하여 넣어주어야 합니다.
- paragraphs에는 retreive한 문서의 key값 10개를 넣어주세요. 반드시 영어로 넣어주어야 인식합니다.
- decomposition은 SARI 측정을 위한 것으로, 현재 지원하지 않습니다. 비워두어도 괜찮습니다.

### 평가 실행법

아래의 코드를 실행합니다.

```shell
python3 evaluate.py --pred=<prediction 파일 경로>
```

## 평가 결과

### gold paragraph

- 정답 paragraph를 모두 주어 준 상태에서 평가를 진행한 결과입니다.

#### prompt Engineering 없는 경우

| Model         | Accuracy |
|---------------|:--------:|
| gpt-3.5-turbo | 0.62857  |

### retriever test

- retriever의 성능을 측정하였습니다

|             Model              | Recall@10 |
|:------------------------------:|:---------:|
|              BM25              | 0.603469  |
|      openai embedding DPR      | 0.408843  |
| openai embedding DPR with HyDE | 0.498469  |
|     KoSimCSE embedding DPR     | 0.1129591 |


## 지원하는 메트릭

현재 Recall@10과 Accuracy 메트릭을 제공하고 있습니다.

## 출처

- 여기서 쓰인 Ko-StrategyQA 데이터는 EDAI에서 직접 제작하였으며 [허깅페이스](https://huggingface.co/datasets/vkehfdl1/Ko-StrategyQA)에 공개되어 있으며,
  자유롭게 사용하실 수 있습니다.
- [원본 strategyQA](https://allenai.org/data/strategyqa)
- [strategyQA Evaluator](https://github.com/allenai/strategyqa-evaluator/tree/main)
