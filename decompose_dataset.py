import torch
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import random

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 실제 모델 경로
device = "cuda:0"  # GPU 지정

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": device}
)
model.eval()

def generate_text(
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.5,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    stop_token: str = None
) -> str:
    """
    단순 텍스트 생성 함수
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # stop_token이 있으면 해당 부분까지로 잘라낸다
    if stop_token and stop_token in decoded:
        decoded = decoded.split(stop_token)[0]

    # 프롬프트 부분 제거(모델이 프롬프트를 그대로 복붙하기도 함)
    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]

    return decoded.strip()

import re

def extract_boxed_answer(solution: str) -> str:
    """
    solution 문자열에서 \boxed{ ... } 부분을 찾아 반환.
    여러 군데 있으면 첫 번째를 반환.
    """
    pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(pattern, solution)
    if match:
        return match.group(1).strip()
    return "No match found"


def hf_chat(prompt, max_new_tokens=256, temperature=0.5):
    """단순 헬퍼 함수"""
    return generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature
    )

import json
import re

def extract_last_json_block(text: str) -> str:
    """
    text 내에서 { ... } 형태의 JSON 블록이 여러 번 있을 수 있으니,
    '가장 마지막'에 있는 { ... } 구간을 추출해 반환.
    찾지 못하면 None 반환.
    """
    matches = list(re.finditer(r'\{.*?\}', text, re.DOTALL))
    if not matches:
        return None
    last_match = matches[-1]
    return text[last_match.start(): last_match.end()]

def parse_json_from_text(raw_text: str, max_retries=1):
    """
    LLM 출력에서 JSON 파싱을 시도:
    1) 가장 마지막 JSON 블록을 찾는다.
    2) json.loads로 파싱한다. (실패 시 None)
    3) 필요하다면 재시도를 증가 가능
    """
    for _ in range(max_retries):
        candidate = extract_last_json_block(raw_text)
        if not candidate:
            return None
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # 깨진 JSON이면 None
            pass
    return None

def call_llm_and_parse_json(prompt: str, 
                            max_new_tokens=256, 
                            temperature=0.3, 
                            max_attempts=3):
    """
    LLM에 prompt를 주고, '마지막 JSON 블록'을 추출해 파싱.
    - max_attempts번 재시도 가능
    - 성공하면 JSON dict 리턴, 실패하면 None
    """
    for attempt in range(1, max_attempts+1):
        raw_output = generate_text(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        parsed = parse_json_from_text(raw_output, max_retries=1)
        if parsed is not None:
            return parsed
    return None

def extract_conditions_objective(problem: str) -> dict:
    """
    LLM에게 '문제 -> 조건과 목적'을 뽑아내도록 요청.
    모델 출력에서 마지막 JSON만 추출 후 파싱.
    """
    # 프롬프트 설계 (예시)
    prompt = f"""You are a helpful assistant. 
                Given the following MATH problem, extract:

                1) "conditions": a list of direct facts stated in the problem
                2) "objective": the main question or goal

                Return ONLY valid JSON. Example:

                {{
                "conditions": [
                    "Louis earns $1200 base monthly salary",
                    "He earns 5% commission on sales",
                    "This month, sales are $25000"
                ],
                "objective": "Find the total monthly earning of Louis."
                }}

                Problem:
                {problem}

                Now respond with JSON only. No extra text.
                """

    raw_output = generate_text(prompt, max_new_tokens=512, temperature=0.3)
    parsed = parse_json_from_text(raw_output)
    print(parsed)
    if not parsed:
        # fallback
        parsed = {
            "conditions": [],
            "objective": ""
        }
    return parsed



def judge_condition(problem_text: str, condition: str):
    prompt = f"""You are Judge.
                You have a math problem and a candidate condition.
                Do NOT vary or create condition. You must only evaluate the given condition against the problem.
                
                Example of wrong response:
                'The equation y = 8x^2 + 5x + 1 represents a parabola'
                -> {{'condition': 'The equation y = 8x^2 + 5x + 1 represents a hyperbola', 'valid': 'False'}}

                Problem:
                {problem_text}

                Condition:
                {condition}
                
                Determine if this condition are valid.
                Output ONLY valid JSON in the format:
                {{
                "condition": {condition},
                "valid": true or false
                }}

                No other commentary or explanation. 
                you must evaluate true or false for ++given condition++, still output JSON format only.
                """
    data = call_llm_and_parse_json(prompt)
    print(data)
    if not data:
        # fallback
        return {"condition": condition, "valid": False}
    return data

def judge_can_solve(conditions: list, objective: str):
    cond_str = "\n".join(f"- {c}" for c in conditions)
    prompt = f"""You are Judge.
                We have the following valid conditions:
                {cond_str}

                Objective:
                {objective}

                Determine if these conditions are sufficient to solve the objective.
                Output ONLY valid JSON in the format:
                {{
                "can_solve": true or false
                }}

                No other commentary or explanation. 
                If you must guess, guess true or false, but still output JSON only.
                """
    data = call_llm_and_parse_json(prompt, max_new_tokens=256, temperature=0.3, max_attempts=3)
    print(data)
    if not data or "can_solve" not in data:
        return {"can_solve": False}
    return data

def executor_generate_steps(conditions: list, solution_text: str):
    cond_str = "\n".join(f"- {c}" for c in conditions)
    prompt = f"""You are Executor.
                Below are some valid conditions and a reference solution from a math dataset.
                You must produce a step-by-step explanation to solve the problem, in JSON.

                Return ONLY valid JSON in this format:
                {{
                "steps": [
                    "Step 1: ...",
                    "Step 2: ...",
                    ...
                ]
                }}

                No extra commentary or explanation outside of the JSON.

                Valid Conditions:
                {cond_str}

                Reference Solution:
                {solution_text}
                Only reference the 'reference solution', don't create conditions via the 'reference solution' that don't exist. 

                No other commentary or explanation. 
                If you must guess, guess true or false, but still output JSON only.
                """
    data = call_llm_and_parse_json(prompt, max_new_tokens=512, temperature=0.3, max_attempts=3)
    print(data)
    if not data or "steps" not in data:
        return {"steps": [], "output": "\\boxed{NO_ANSWER}"}
    # 여기선 output 키가 없다면 수작업으로 \boxed{} 만들거나 생략
    return data

def clean_judge_condition_validities(condition_validities):
    """
    condition_validities 배열을 순회하며:
      - "condition" 키를 무조건 변경 (모든 항목에서 첫 번째 값을 "condition"으로 사용)
      - "valid" 값 처리 로직은 기존과 동일하게 유지
    """
    cleaned = []

    for item in condition_validities:
        # 1️⃣ "condition" 키 강제 변경 (모든 키 중 첫 번째 값 사용)
        condition = next(iter(item.values()), "")  # 첫 번째 값 가져오기

        # 2️⃣ "valid" 값 처리 (기존 코드 유지)
        val = item.get("valid", None)

        if isinstance(val, bool):
            valid = val  # 이미 bool이면 그대로 사용
        elif isinstance(val, dict):
            valid = val.get("valid", None) if isinstance(val.get("valid"), bool) else None
        else:
            valid = None  # 그 외 값은 무시

        if valid is not None:  # valid 값이 정상적인 경우만 추가
            cleaned.append({
                "condition": str(condition).strip(),  # 무조건 문자열로 변환
                "valid": valid
            })

    return cleaned



def clean_judge_can_solve(can_solve_obj):
    """
    can_solve가 다음과 같이 중첩될 수 있음:
    { "can_solve": { "can_solve": true } }
    여기서 가장 깊은 true/false를 찾아서 반환.
    """
    # 재귀적으로 파고들어, bool이 나오면 반환
    if isinstance(can_solve_obj, bool):
        return can_solve_obj
    if isinstance(can_solve_obj, dict):
        for k, v in can_solve_obj.items():
            cleaned = clean_judge_can_solve(v)  # 재귀
            if isinstance(cleaned, bool):
                return cleaned
    # 기본값
    return False

def append_dataset_answer(executor_result: dict, dataset_solution: str):
    """
    executor_result = {"steps": [...], "output": "..."} 등
    dataset_solution에서 \boxed{} 구문을 찾아, executor_result에 추가
    """
    final_ans = extract_boxed_answer(dataset_solution)  # "\\boxed{...}" 내부 추출
    if final_ans == "No match found":
        return executor_result

    # 원하는 필드 이름(예: "official_boxed_answer")을 새로 추가
    executor_result["official_boxed_answer"] = f"\\boxed{{{final_ans}}}"

    # 혹은 기존 output에 덧붙일 수도 있음:
    # executor_result["output"] += f" (Official: \\boxed{{{final_ans}}})"

    return executor_result

import os
import json
from tqdm import tqdm

DATASET_DIR = "MATH/train"
OUTPUT_DIR = "MATH/processed_data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_math_dataset(dataset_dir, output_dir, limit=None):
    for subfolder in os.listdir(dataset_dir):
        subfolder_path = os.path.join(dataset_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        out_subfolder = os.path.join(output_dir, subfolder)
        os.makedirs(out_subfolder, exist_ok=True)

        files = [f for f in os.listdir(subfolder_path) if f.endswith(".json")]
        random.shuffle(files)

        print(f"[INFO] Processing '{subfolder}', total={len(files)}")

        for count, filename in tqdm(enumerate(files[:limit])):
            file_path = os.path.join(subfolder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except:
                continue

            problem = data.get("problem", "")
            solution = data.get("solution", "")

            base_name = os.path.splitext(filename)[0]

            # 1) Thinker
            thinker_result = extract_conditions_objective(problem)
            conditions = thinker_result.get("conditions", [])
            objective = thinker_result.get("objective", "")

            # Thinker 결과 저장
            thinker_data = {
                "role": "thinker",
                "task_type": "extract_conditions_objective",
                "input": problem,
                "output": thinker_result
            }
            
            # ============ (1) 조건이 비어있으면 이후 단계를 스킵 ===========
            if not conditions:
                continue  # 다음 파일로 넘어감
                
            # 2) Judge
            judge_condition_list = []
            valid_conditions = []

            # (a) condition_validity
            for cond in conditions:
                single_judge_res = judge_condition(problem, cond)  
                # 예: {"condition": cond, "valid": True} or {"valid": { ... }}
                judge_condition_list.append(single_judge_res)

            # 후처리
            cleaned_condition_validities = clean_judge_condition_validities(judge_condition_list)

            # valid_conditions
            for citem in cleaned_condition_validities:
                if citem["valid"] is True:
                    valid_conditions.append(citem["condition"])

            # (b) can_solve
            raw_can_solve_obj = judge_can_solve(valid_conditions, objective)  
            # 예: {"can_solve": True} or 중첩
            raw_val = raw_can_solve_obj.get("can_solve", False)  
            can_solve_bool = clean_judge_can_solve(raw_val)

            judge_data = {
                "role": "judge",
                "task_type": "condition_validity + can_solve",
                "input": {
                    "problem": problem,
                    "conditions": conditions,
                    "objective": objective
                },
                "output": {
                    "condition_validities": [
                        # 예: {"condition": "...", "valid": true/false}
                        # (cleaned)
                        {"condition": c["condition"], "valid": c["valid"]} 
                        for c in cleaned_condition_validities
                    ],
                    "can_solve": can_solve_bool
                }
            }
            
            if not can_solve_bool:
                continue  # 다음 파일로 넘어감
            # ============ (2) Executor ===========
            # valid_conditions + solution -> executor
            executor_result = executor_generate_steps(valid_conditions, solution)
            # 예: {"steps": [...], "output": "..."}
            # -> 실제 데이터셋 정답 \boxed{} 추가
            executor_result = append_dataset_answer(executor_result, solution)

            executor_data = {
                "role": "executor",
                "task_type": "generate_steps",
                "input": {
                    "conditions": valid_conditions,
                    "solution": solution
                },
                "output": executor_result
            }
            
            thinker_file = os.path.join(out_subfolder, f"{base_name}_thinker.json")
            with open(thinker_file, "w", encoding="utf-8") as ft:
                json.dump(thinker_data, ft, indent=2, ensure_ascii=False)

            judge_file = os.path.join(out_subfolder, f"{base_name}_judge.json")
            with open(judge_file, "w", encoding="utf-8") as fj:
                json.dump(judge_data, fj, indent=2, ensure_ascii=False)

            executor_file = os.path.join(out_subfolder, f"{base_name}_executor.json")
            with open(executor_file, "w", encoding="utf-8") as fx:
                json.dump(executor_data, fx, indent=2, ensure_ascii=False)

        print(f"[DONE] {subfolder} => {out_subfolder}")


# 실행 예시
if __name__ == "__main__":
    process_math_dataset(DATASET_DIR, OUTPUT_DIR, limit=500)
    print("All done!")