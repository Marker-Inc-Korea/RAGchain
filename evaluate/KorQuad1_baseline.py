import json
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain import LLMChain
from tqdm import tqdm

PROMPT_TEMPLATE = """주어진 정보를 바탕으로 질문에 답하세요. 답을 모른다면 답을 지어내려고 하지 말고 모른다고 답하세요. 
    질문 이외의 상관 없는 답변을 하지 마세요. 반드시 한국어로 답변하세요. 단답형으로 답변하세요.
    
    {context}
    
    질문: {question}
    한국어 답변:"""
STOP_WORDS = [".", "#", "답변:", "응답:", "\n", "맥락", "?"]

def main():
    with open("../../KorQuad1.0/Dev-set/KorQuAD_v1.0_dev.json") as f:
        dev_set = json.load(f)

    model = CTransformers(model="vkehfdl1/KoAlpaca-Polyglot-12.8b-ggml",
                          model_file="KoAlpaca-Polyglot-12.8b-ggml-model-q5_0.bin",
                          model_type="gpt_neox")
    answer_result = dict()
    for i in tqdm(range(30)):
        for data in dev_set['data'][i]['paragraphs']:
            context = data["context"]
            for qa in data['qas']:
                question = qa["question"]
                id = qa["id"]
                answer = get_answer(model, question, context)
                for stop_word in STOP_WORDS:
                    if stop_word in answer:
                        temp_ans = answer.split(stop_word)[0]
                        if temp_ans:
                            answer = temp_ans
                answer_result[id] = answer


    # write json file to answer_result.json
    with open("korQuad1_30_no_vector_db_koAlpaca_polyglot_12.8b.json", "w") as f:
        json.dump(answer_result, f)


def get_answer(model, question, context):
    prompt = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = LLMChain(llm=model, prompt=prompt)
    return chain.run(question=question, context=context)


if __name__ == "__main__":
    main()
