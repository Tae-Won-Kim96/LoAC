# 대신, 우리가 만든 hf_model.py import
from utils.hf_model import (
    # hf_chat,
    hf_chat_thinker,
    hf_chat_judge,
    hf_chat_executor
)

def generate_from_thinker(prompts, max_tokens=512, temperature=0.7):
    """
    Use the local LLaMA model to generate responses for the Thinker role.
    """
    responses = []
    for prompt in prompts:
        message = prompt["content"]

        try:
            # 로컬 LLaMA 모델을 통해 응답 생성
            response = hf_chat_thinker(message, max_tokens=max_tokens, temperature=temperature)
            responses.append(response)
        except Exception as e:
            print(f"[Thinker] Error occurred: {e}")
            responses.append("I need to rethink this problem.")  # 기본 응답 설정

    return responses

def generate_from_judge(prompts, max_tokens=512, temperature=0.7):
    """
    Use the local LLaMA model to generate responses for the Judge role.
    """
    responses = []
    for prompt in prompts:
        message = prompt["content"]

        try:
            # 로컬 LLaMA 모델을 통해 응답 생성
            response = hf_chat_judge(message, max_tokens=max_tokens, temperature=temperature)
            responses.append(response)
        except Exception as e:
            print(f"[Judge] Error occurred: {e}")
            responses.append("False")  # 기본 응답 설정

    return responses

def generate_from_executor(prompts, max_tokens=512, temperature=0.7):
    """
    Use the local LLaMA model to generate responses for the Executor role.
    """
    responses = []
    for prompt in prompts:
        message = prompt["content"]

        try:
            # 로컬 LLaMA 모델을 통해 응답 생성
            response = hf_chat_executor(message, max_tokens=max_tokens, temperature=temperature)
            responses.append(response)
        except Exception as e:
            print(f"[Executor] Error occurred: {e}")
            responses.append("False")  # 기본 응답 설정

    return responses

# def chat_gpt(prompt):
#     """
#     OpenAI chat_gpt 대체 -> Llama 모델로 단순 text를 생성
#     """
#     # 만약 messages가 아니라 단순 스트링이라면 아래처럼 호출
#     messages = [
#         {"role": "user", "content": prompt}
#     ]
#     response = hf_chat(messages, max_new_tokens=512)  # 원하는 max_new_tokens
#     return response

# def generate_from_GPT(prompts, max_tokens=800, model=None, temperature=0.7, n=1):
#     """
#     기존에 prompts(list of dict)로 chat.completions를 받았던 함수를 
#     hf_chat로 호출하도록 수정
#     """
#     # 1) prompts가 이미 아래 형태라고 가정:
#     #    [ {"role":"system"/"user"/"assistant", "content": ...}, ... ]
#     #    n=1 이면 1개, n>1 이면 여러개 시도 -> 아래는 간단히 1개만 반환하는 형태
#     #    (필요하다면 loop 혹은 beam search 로직 추가)
    
#     # 만약 max_tokens를 max_new_tokens로 맞춰주고 싶으면:
#     response_text = hf_chat(
#         messages=prompts,
#         max_new_tokens=max_tokens,
#         temperature=temperature
#     )
    
#     # OpenAI API에서는 여러 response가 나오지만, 
#     # 여기서는 하나만 반환할 수 있도록 간소화 (n=1)
#     # 필요하면 여러번 호출하는 로직 추가
#     # return 형태 맞추기 위해, 아래처럼 구조를 흉내낼 수도 있습니다:
#     result = [
#         {
#             "index": 0,
#             "message": {
#                 "role": "assistant",
#                 "content": response_text
#             },
#             "finish_reason": "stop"
#         }
#     ]
#     return result


# def Judge_if_got_Answer_from_GPT(prompts, max_tokens=800, model=None, temperature=0.7, n=1):
#     """
#     기존 Judge 함수 대체 -> 간단히 hf_chat로
#     """
#     return hf_chat_judge(prompts, max_new_tokens=max_tokens, temperature=temperature)


# def Find_Answer_from_GPT(prompts, max_tokens=800, model=None, temperature=0.7, n=1):
#     """
#     마찬가지로 hf_chat 사용
#     """
#     return hf_chat_executor(prompts, max_new_tokens=max_tokens, temperature=temperature)

# import code
# from openai import OpenAI
# import backoff 


# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key="",
# ) # Input your own API-Key

# def chat_gpt(prompt):
#     response = client.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message.content.strip()


# def generate_from_GPT(prompts, max_tokens, model="gpt-4-1106-preview", temperature=0.7, n=3):
#     """
#     Generate answer from GPT model with the given prompt.
#     input:
#         @max_tokens: the maximum number of tokens to generate; in this project, it is 8000 - len(fortran_code)
#         @n: the number of samples to return
#     return: a list of #n generated_ans when no error occurs, otherwise None

#     return example (n=3):
#         [
#         {
#         "index": 0,
#         "message": {
#             "role": "assistant",
#             "content": "The meaning of life is subjective and can vary greatly"
#         },
#         "finish_reason": "length"
#         },
#         {
#         "index": 1,
#         "message": {
#             "role": "assistant",
#             "content": "As an AI, I don't have personal beliefs"
#         },
#         "finish_reason": "length"
#         },
#         {
#         "index": 2,
#         "message": {
#             "role": "assistant",
#             "content": "The meaning of life is subjective and can vary greatly"
#         },
#         "finish_reason": "length"
#         }
#     ]
#     """
#     import openai
#     openai.api_key = "" # TODO

#     try:
#         result = completions_with_backoff(
#             model = model, 
#             messages = prompts, 
#             temperature = temperature, 
#             max_tokens = max_tokens, 
#             n = n
#         )

#         generated_ans = result["choices"]
#         return generated_ans
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None


# def Judge_if_got_Answer_from_GPT(prompts, max_tokens, model="gpt-4-1106-preview", temperature=0.7, n=1):
#     """
#     Generate answer from GPT model with the given prompt.
#     input:
#         @max_tokens: the maximum number of tokens to generate; in this project, it is 8000 - len(fortran_code)
#         @n: the number of samples to return
#     return: a list of #n generated_ans when no error occurs, otherwise None

#     return example (n=3):
#     """
#     try:
#         response = client.chat.completions.create(
#             model=model,
#             messages=prompts,
#             max_tokens = max_tokens,
#             temperature = temperature,
#             n = n
#         )
#         return response.choices[0].message.content.strip()
    
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         return None
    
    
# def Find_Answer_from_GPT(prompts, max_tokens, model="gpt-4-1106-preview", temperature=0.7, n=1):
def Find_Answer_from_GPT(prompts, max_new_tokens=1024, role='executor', temperature=0.7, n=1):
    """
    Generate answer from GPT model with the given prompt.
    input:
        @max_tokens: the maximum number of tokens to generate; in this project, it is 8000 - len(fortran_code)
        @n: the number of samples to return
    return: a list of #n generated_ans when no error occurs, otherwise None

    return example (n=3):
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=prompts,
            max_tokens = max_tokens,
            temperature = temperature,
            n = n
        )
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
    responses = []
    for prompt in prompts:
        message = prompt["content"]

        try:
            # 로컬 LLaMA 모델을 통해 응답 생성
            response = hf_chat_executor(message, max_tokens=max_tokens, temperature=temperature)
            responses.append(response)
        except Exception as e:
            print(f"[Executor] Error occurred: {e}")
            responses.append("False")  # 기본 응답 설정

    return responses
    """
    try:
        # 가장 최근 user 입력만 추출
        user_input = prompts[-1]["content"]

        # 역할에 따라 적절한 모델 사용
        if role == "thinker":
            response = hf_chat_thinker(user_input, max_new_tokens=max_tokens, temperature=temperature)
        elif role == "judge":
            response = hf_chat_judge(user_input, max_new_tokens=max_tokens, temperature=temperature)
        elif role == "executor":
            response = hf_chat_executor(user_input, max_new_tokens=max_tokens, temperature=temperature)
        else:
            raise ValueError("Invalid role. Choose from 'thinker', 'judge', 'executor'.")

        return response if n == 1 else [response]  # n=1이면 문자열, n>1이면 리스트 반환

    except Exception as e:
        print(f"[ERROR] LLaMA model execution failed: {e}")
        return None
