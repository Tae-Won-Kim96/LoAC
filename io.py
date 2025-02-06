import torch
import jsonlines, random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_answer(prompt, model, tokenizer, device, max_length=512):
    """모델이 수학 문제의 답변을 생성하는 함수"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length, num_return_sequences=1,pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output[len(prompt):].strip()

def extract_boxed_answer(answer, flag=False):
    """'####' 이후의 답을 추출하는 함수"""
    if "####" not in answer:
        return "No Answer(lack of TL)"  # "####"이 없으면 기본값 반환
    
    extracted = answer.split("####")[-1]
    print(f'extracted answer: {extracted}')
    extracted = extracted.strip()  # "####" 이후의 텍스트 추출
        
    if not extracted:
        return "No Answer"  # 추출된 값이 빈 문자열이면 기본값 반환
    
    return extracted # 불필요한 문자 제거
import re

def extract_boxed_answer(answer, flag=False):
    if "####" not in answer:
        return "No Answer (lack of TL)" 

    matches = re.findall(r'####\s*([-+]?\d*\.?\d+)', answer)
    
    if matches:
        if flag:
            print(f'CoT answer: {matches[-1]}')
        else:
            print(f'I/O answer: {matches[-1]}')
        return matches[-1]

    return "No Answer" 

def process_item(item, base_model, cot_model, tokenizer):
    """하나의 문제를 처리하는 함수 (단일 GPU 실행)"""
    problem = item["question"]
    solution = item["answer"]

    if not problem or not solution:
        return None

    base_input = base_prompt.format(problem=problem)
    cot_input = cot_prompt.format(problem=problem)

    # ✅ 단일 GPU에서 실행
    base_answer = generate_answer(base_input, base_model, tokenizer, "cuda:0", 256)
    print(base_answer.split('\n')[-1])
    cot_answer = generate_answer(cot_input, cot_model, tokenizer, "cuda:1", 512)
    print(cot_answer.split('\n')[-1])
    base_answer = extract_boxed_answer(base_answer)
    cot_answer = extract_boxed_answer(cot_answer, True)
    extracted_solution = solution.split("####")[-1].strip()
    print(f"Answer: {extracted_solution}\nIO: {base_answer}, CoT: {cot_answer}")

    try:
        correct_base = 1 if float(base_answer) == float(extracted_solution) else 0
    except ValueError:
        correct_base = 0

    try:
        correct_cot = 1 if float(cot_answer) == float(extracted_solution) else 0
    except ValueError:
        correct_cot = 0

    return {
        "problem": problem,
        "solution": solution,
        "correct_base": correct_base,
        "correct_cot": correct_cot
    }

def evaluate_dataset_by_topic_with_prompt(folder_path, base_prompt, cot_prompt, model_name, limit=100):
    """단일 GPU에서 실행하는 함수"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ✅ 모델을 단일 GPU에서 로드
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:0")
    cot_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda:1")

    # JSON 데이터 로드 및 제한 적용
    data = []
    with jsonlines.open("data/test.jsonl", "r") as f:
        for line in f:
            data.append(line)

    random.shuffle(data)
    data = data[:limit]
    results = []
    for item in tqdm(data, total=limit):
        result = process_item(item, base_model, cot_model, tokenizer)
        if result:
            results.append(result)

    # 결과 집계
    correct_base = sum(r["correct_base"] for r in results if r)
    correct_cot = sum(r["correct_cot"] for r in results if r)
    total = len(results)

    accuracy_base = (correct_base / total) * 100 if total > 0 else 0
    accuracy_cot = (correct_cot / total) * 100 if total > 0 else 0

    topic_results = {
        "correct_base": correct_base,
        "correct_cot": correct_cot,
        "total": total,
        "accuracy_base": accuracy_base,
        "accuracy_cot": accuracy_cot
    }

    print("\nFinal Results:")
    print(f"Base Accuracy = {topic_results['accuracy_base']:.2f}%, CoT Accuracy = {topic_results['accuracy_cot']:.2f}% ({correct_base}/{total} base, {correct_cot}/{total} CoT)")

    # 결과 저장
    output_txt_file = "GSM8K_Results.txt"
    with open(output_txt_file, "w") as f:
        result_line = (
            f"Base Accuracy = {accuracy_base:.2f}%, "
            f"CoT Accuracy = {accuracy_cot:.2f}% "
            f"({correct_base}/{total} base, {correct_cot}/{total} CoT)\n"
        )
        f.write(result_line)

    print(f"Results saved to {output_txt_file}")

if __name__ == "__main__":
    folder_path = "data/test"
    model_name = "meta-llama/Llama-3.1-8B-Instruct"

    base_prompt = """{problem}
Answer only with a number with annotation form on the LAST LINE and do not add any other characters.
Just end your answer with '#### number'.

Example:
Natalia sold paperclips to 48 of her friends in April, and then sold half that number in May.
How many clips did Natalia sell in total in April and May?

#### 72
"""

    cot_prompt = """Solve the following math word problem step-by-step: {problem}    
Answer only with a number with annotation form on the LAST LINE and do not add any other characters.
Just end your answer with '#### number'.

Example:
Natalia sold paperclips to 48 of her friends in April, and then sold half that number in May.
How many clips did Natalia sell in total in April and May?

Step 1: Number of clips she sold in April: 48
Step 2: Calculate how many clips Natalia sold in May: 48 / 2 = 24
Step 3: Calculate the total number of sales: 48 + 24 = 72

#### 72
"""


    evaluate_dataset_by_topic_with_prompt(folder_path, base_prompt, cot_prompt, model_name, limit=199)


#     base_prompt = """{problem}
# The answer should be provided on the last line and be followed immediately by ####
# Do not generate any words after your answer, and in the last line make your answer consist of only final answer followed by ####.
# Example: 
# Q: Natalia sold paperclips to 48 of her friends in April, and then sold half that number in May. How many clips did Natalia sell in total in April and May?
# Last line: #### 72
#     """

#     cot_prompt = """Answer the following question by identifying intermediate steps: {problem}
# The answer should be provided on the last line and be followed immediately by ####
# Do not generate any words after your answer, and in the last line make your answer consist of only final answer followed by ####. 
# Final line don't contain any steps.
# Example: 
# Q: Natalia sold paperclips to 48 of her friends in April, and then sold half that number in May. How many clips did Natalia sell in total in April and May?

# step 1: Number of clips she sold in April: <48>
# step 2: Calculate how many clips Natalia sold in May: <48/2=24>
# step 3: Calculate the total number of sales: <48+24=72>

# Last line: \n#### 72
#     """