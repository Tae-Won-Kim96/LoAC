def generate_thinker_data(problem):
    """
    Thinker: 문제에서 조건(conditions)과 목표(objective)를 JSON으로 추출.
    """
    prompt = f"""
    You are an AI assistant tasked with analyzing math problems. 

    Given the following math problem:
    "{problem}"

    1. Identify and list all conditions that can be directly extracted from the problem.  
    2. Define the main objective of the problem in one sentence.

    Format your response as follows:
    Conditions:
    1. ...
    2. ...
    3. ...
    Objective:
    ...
    """
    
    response = request("thinker", prompt)
    parsed_response = fix_invalid_json(response)

    if parsed_response:
        print("=== Thinker Response ===\n", parsed_response, "\n========================\n")
        return parsed_response
    else:
        print("[ERROR] Thinker response parsing failed.")
        return None

def generate_judge_data(problem, objective, conditions):
    """
    Judge: 문제(Problem)와 조건(Condition)을 받아 각 조건의 유효성(true/false)을 판단.
    """
    judged_conditions = []

    for condition in conditions:
        prompt = f"""
        You are an AI judge tasked with verifying whether extracted conditions from a math problem are correct.

        Math problem:
        "{problem}"

        Conditions:
        {condition}

        1. Check each condition and mark it as True (valid) or False (invalid).
        2. Determine if these conditions are sufficient to solve the problem.

        Format:
        {
        "valid_conditions":
            {{"statement": "...", "valid": true/false}}
        }
        """

        response = request("judge", prompt)
        parsed_response = fix_invalid_json(response)

        if isinstance(parsed_response, dict) and "condition" in parsed_response and "valid" in parsed_response:
            judged_conditions.append(parsed_response)
        else:
            print("[WARN] Judge returned invalid response.")
            print(f"[DEBUG] Raw response: {response}")

    return judged_conditions


You are an AI assistant tasked with solving math problems using given conditions.

Math problem:
"{problem}"

Verified Conditions:
{valid_conditions}

Objective:
{objective}

1. Generate a step-by-step solution using the given conditions.
2. Provide the final answer in LaTeX format using \boxed{}.

Format:
{
  "steps": [
    "Step 1: ...",
    "Step 2: ..."
  ],
  "solution": "\\boxed{...}"
}
