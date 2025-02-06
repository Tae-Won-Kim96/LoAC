# utils/hf_model.py (새로 만들거나 기존 파일에 추가)

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
# from peft import PeftModel
# import textwrap

# # 사용할 모델 이름 (HF Hub)
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 실제로 존재하는 모델 경로로 바꾸세요

# # -----------------------------------------------------------------
# # (1) Thinker 모델: GPU 0
# # -----------------------------------------------------------------
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.eos_token_id

# def generate_text(model, tokenizer, prompt, 
#                   max_new_tokens=256,
#                   temperature=0.5,
#                   top_p=0.9,
#                   repetition_penalty=1.1,
#                   stop_token=None):
#     """
#     LLaMA 기반 모델에서 텍스트 생성
#     """

#     # (1) pad_token_id 설정
#     if tokenizer.pad_token_id is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.pad_token_id = tokenizer.eos_token_id

#     # (2) 입력 처리 + `attention_mask` 추가
#     inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
#     attention_mask = inputs.attention_mask  # 추가됨

#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids=inputs.input_ids,
#             attention_mask=attention_mask,  # `attention_mask` 추가
#             max_new_tokens=max_new_tokens,
#             temperature=temperature,
#             top_p=top_p,
#             repetition_penalty=repetition_penalty,
#             do_sample=True,
#             eos_token_id=tokenizer.eos_token_id,
#             pad_token_id=tokenizer.pad_token_id
#         )

#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     if stop_token and stop_token in decoded:
#         decoded = decoded.split(stop_token)[0]

#     if decoded.startswith(prompt):
#         decoded = decoded[len(prompt):]

#     return decoded.strip()

# model_thinker = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map={"": "cuda:0"}  # Judge -> GPU1
# )


# def hf_chat_thinker(prompt, max_new_tokens=256, temperature=0.5):
#     p_token = len(prompt.split(' '))
    
#     output = generate_text(model_thinker, tokenizer, prompt, max_new_tokens=p_token+max_new_tokens)
#     return output


# # -----------------------------------------------------------------
# # (2) Judge 모델: GPU 1
# # -----------------------------------------------------------------
# # tokenizer_judge = AutoTokenizer.from_pretrained(MODEL_NAME)
# # if tokenizer_judge.pad_token_id is None:
# #     tokenizer_judge.pad_token_id = tokenizer_judge.eos_token_id

# model_judge = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map={"": "cuda:2"}  # Judge -> GPU1
# )
# def hf_chat_judge(prompt, max_new_tokens=256, temperature=0.5):
#     p_token = len(prompt.split(' '))
#     output = generate_text(model_judge, tokenizer, prompt, max_new_tokens=p_token+max_new_tokens)
#     return output


# # -----------------------------------------------------------------
# # (3) Executor 모델: GPU 2
# # -----------------------------------------------------------------
# # tokenizer_executor = AutoTokenizer.from_pretrained(MODEL_NAME)
# # if tokenizer_executor.pad_token_id is None:
# #     tokenizer_executor.pad_token_id = tokenizer_executor.eos_token_id

# model_executor = AutoModelForCausalLM.from_pretrained(
#     MODEL_NAME,
#     torch_dtype=torch.float16,
#     device_map={"": "cuda:1"}  # Executor -> GPU2
# )


# def hf_chat_executor(prompt, max_new_tokens=1024, temperature=0.5):
#     p_token = len(prompt.split(' '))
#     output = generate_text(model_executor, tokenizer, prompt, max_new_tokens=p_token+max_new_tokens)
#     return output

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import textwrap

# 사용할 모델 이름 (HF Hub)
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # 실제로 존재하는 모델 경로로 바꾸세요

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 모델을 GPU에 로드 (기본 모델)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map={"": "cuda:1"}  # GPU 자동 분배
)

LORA_FOLDER_THINKER = "lora_weights/lora_thinker_weights"
LORA_FOLDER_JUDGE   = "lora_weights/lora_judge_weights"
LORA_FOLDER_EXEC    = "lora_weights/lora_executor_weights"

def load_lora_for_role(base_model, lora_path, device="cuda:1"):
    print(f"Applying LoRA from {lora_path} ...")
    lora_model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.float16
    ).to(device)
    lora_model.eval()
    return lora_model

model_thinker = load_lora_for_role(base_model, LORA_FOLDER_THINKER)
model_judge   = load_lora_for_role(base_model, LORA_FOLDER_JUDGE)
model_executor = load_lora_for_role(base_model, LORA_FOLDER_EXEC)


def generate_text(model, tokenizer, prompt, 
                  max_new_tokens=256,
                  temperature=0.5,
                  top_p=0.9,
                  repetition_penalty=1.1,
                  stop_token=None):
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    attention_mask = inputs.attention_mask  # 추가됨

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=attention_mask,  # `attention_mask` 추가
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if stop_token and stop_token in decoded:
        decoded = decoded.split(stop_token)[0]

    if decoded.startswith(prompt):
        decoded = decoded[len(prompt):]

    return decoded.strip()

def hf_chat_thinker(prompt, max_new_tokens=256, temperature=0.5):
    p_token = len(prompt.split(' '))
    
    output = generate_text(model_thinker, tokenizer, prompt, max_new_tokens=p_token+max_new_tokens)
    return output

def hf_chat_judge(prompt, max_new_tokens=256, temperature=0.5):
    p_token = len(prompt.split(' '))
    output = generate_text(model_judge, tokenizer, prompt, max_new_tokens=p_token+max_new_tokens)
    return output

def hf_chat_executor(prompt, max_new_tokens=1024, temperature=0.5):
    p_token = len(prompt.split(' '))
    output = generate_text(model_executor, tokenizer, prompt, max_new_tokens=p_token+max_new_tokens)
    return output
