import torch
import jsonlines, random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

LORA_FOLDER = "LoRA/lora_llama3_gsm8k"

def load_lora_for_role(base_model, lora_path, device="cuda:1"):
    print(f"Applying LoRA from {lora_path} ...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    ).to(device)
    lora_model.eval()
    return lora_model

def generate_answer(prompt, model, tokenizer, device, max_length=512):
    """모델이 수학 문제의 답변을 생성하는 함수"""
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
    outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=max_length, num_return_sequences=1,pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output[len(prompt):].strip()
import re

def extract_boxed_answer(answer, flag=False):
    if "####" not in answer:
        return "No Answer (lack of TL)" 

    matches = re.findall(r'####\s*([-+]?\d*\.?\d+)', answer)
    
    if matches:
        return matches[-1]

    return "No Answer" 

def process_item(item, base_model, tokenizer):
    problem = item["question"]
    solution = item["answer"]

    if not problem or not solution:
        return None

    base_input = base_prompt.format(problem=problem)
    base_answer = generate_answer(base_input, base_model, tokenizer, "cuda:1", 256)
    print(base_answer)
    base_answer = extract_boxed_answer(base_answer)
    extracted_solution = solution.split("####")[-1].strip()
    print(f"Answer: {extracted_solution}\nLoRA: {base_answer}")

    try:
        correct_base = 1 if float(base_answer) == float(extracted_solution) else 0
    except ValueError:
        correct_base = 0

    return {
        "problem": problem,
        "solution": solution,
        "correct_base": correct_base,
    }

def evaluate_dataset_by_topic_with_prompt(folder_path, base_prompt, model_name, limit=100):
    tokenizer = AutoTokenizer.from_pretrained(LORA_FOLDER)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    base_model = PeftModel.from_pretrained(model, LORA_FOLDER).to("cuda:1")
    data = []
    with jsonlines.open("data/test.jsonl", "r") as f:
        for line in f:
            data.append(line)

    random.shuffle(data)
    data = data[:limit]
    results = []
    for item in tqdm(data, total=limit):
        result = process_item(item, base_model, tokenizer)
        if result:
            results.append(result)

    correct_base = sum(r["correct_base"] for r in results if r)
    total = len(results)

    accuracy_base = (correct_base / total) * 100 if total > 0 else 0

    topic_results = {
        "correct_base": correct_base,
        "total": total,
        "accuracy_base": accuracy_base,
    }

    print("\nFinal Results:")
    print(f"Base Accuracy = {topic_results['accuracy_base']:.2f}% ({correct_base}/{total} LoRA)")

    # 결과 저장
    output_txt_file = "GSM8K_Results.txt"
    with open(output_txt_file, "w") as f:
        result_line = (
            f"Base Accuracy = {accuracy_base:.2f}%, "
            f"({correct_base}/{total} LoRA\n"
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
answer:{{
...
#### 72
}}
"""

    evaluate_dataset_by_topic_with_prompt(folder_path, base_prompt, model_name, limit=199)