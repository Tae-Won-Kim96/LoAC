# utils/gpt_robots.py

from utils.hf_model import (
    hf_chat_thinker,
    hf_chat_judge,
    hf_chat_executor
)

def generate_from_thinker(prompts, max_tokens=256, model=None, temperature=0.7, n=1):
    """
    Thinker -> GPU0
    기존에는 openai Beta assistant를 썼지만,
    이제 hf_chat_thinker(...) 사용
    """
    # prompts가 list[dict] 형태라면, 한 덩어리의 string으로 합쳐야 함
    conversation = []
    for msg in prompts:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # 간단히 "[User] content" 형태로 누적
        conversation.append(f"[{role.upper()}] {content}")

    final_prompt = "\n".join(conversation)
    return hf_chat_thinker(final_prompt, max_new_tokens=max_tokens, temperature=temperature)


def generate_from_judge(prompts, max_tokens=256, model=None, temperature=0.7, n=1):
    """
    Judge -> GPU1
    """
    conversation = []
    for msg in prompts:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        conversation.append(f"[{role.upper()}] {content}")

    final_prompt = "\n".join(conversation)
    return hf_chat_judge(final_prompt, max_new_tokens=max_tokens, temperature=temperature)


def generate_from_excutor(prompts, max_tokens=256, model=None, temperature=0.7, n=1):
    """
    Executor -> GPU2
    """
    conversation = []
    for msg in prompts:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        conversation.append(f"[{role.upper()}] {content}")

    final_prompt = "\n".join(conversation)
    return hf_chat_executor(final_prompt, max_new_tokens=max_tokens, temperature=temperature)
