import os
import json
import random
import torch
import re

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

from tqdm import tqdm




def escape_braces(text):
    return text.split("####")[-1].strip()

def extract_boxed_answer(answer):
    pattern = r'\\boxed\{(.*)(?=\})'

    matches = re.search(pattern, answer)
    if matches:
        match = matches.group(1)
    else:
        match = "No match found"
    return match

def generate_answer(prompt, model, tokenizer, device):
    # Tokenize input and create attention mask
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,  # Ensure input is padded
        truncation=True,  # Truncate if the input exceeds model's max length
        max_length=1024  # Adjust max length as necessary
    ).to(device)

    # Explicitly pass attention_mask during generation
    outputs = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,  # Include the attention mask
        max_length=256,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id  # Set the padding token
    )

    # Decode and clean the output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Remove the prompt from the output
    answer = decoded_output[len(prompt):].strip()  # Remove the prompt and strip whitespace
    return answer


def evaluate_dataset_by_topic_with_prompt(folder_path, base_prompt, cot_prompt, model_name, limit=100):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load models on separate GPUs
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
        # quantization_config=...  # 필요 시 BitsAndBytesConfig 등 사용 가능
    ).to("cuda:0")
    cot_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
        # quantization_config=w...  # 필요 시 BitsAndBytesConfig 등 사용 가능
    ).to("cuda:1")

    topic_results = {}

    for topic in os.listdir(folder_path):
        topic_path = os.path.join(folder_path, topic)
        if not os.path.isdir(topic_path):
            continue

        print(f"\nEvaluating topic: {topic}")
        topic_files = [os.path.join(topic_path, file) for file in os.listdir(topic_path) if file.endswith('.json')]
        random.shuffle(topic_files)

        correct_base = 0
        correct_cot = 0
        total = 0

        for count, file_path in tqdm(enumerate(topic_files[:limit])):
            with open(file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    problem = data.get("problem")
                    solution = data.get("solution", "")

                    # 디버깅: 문제 텍스트 확인
                    print(file_path)
                    print(f"Raw Problem: {str(problem)}")
                    print(f'Escaped solution: {extract_boxed_answer(solution)}')

                    if problem and solution:
                        # Base Prompt
                        base_input = base_prompt.format(problem=str(problem))
                        base_answer = generate_answer(base_input, base_model, tokenizer, "cuda:0")
                        print(f'base answer:\n\n {base_answer} \n\n')
                        base_answer = extract_boxed_answer(base_answer)
                        print(f"Base Answer: {base_answer}, Solution: {extract_boxed_answer(solution.strip())}")

                        if (base_answer.strip()) == (extract_boxed_answer(solution.strip())):
                            correct_base += 1
                            print("===========================\nGenerating correct answer...\n===========================\n")

                        # CoT Prompt
                        cot_input = cot_prompt.format(problem=problem)
                        cot_answer = generate_answer(cot_input, cot_model, tokenizer, "cuda:1")
                        cot_answer = extract_boxed_answer(cot_answer)
                        print(f"CoT Answer: {cot_answer}, solution: {extract_boxed_answer(solution.strip())}")

                        if (cot_answer.strip()) == (extract_boxed_answer(solution.strip())):
                            correct_cot += 1
                            print("===========================\nGenerating correct answer...\n===========================\n")

                        total += 1
                except json.JSONDecodeError:
                    print(f"Error processing file {file_path}: Invalid JSON format.")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        accuracy_base = (correct_base / total) * 100 if total > 0 else 0
        accuracy_cot = (correct_cot / total) * 100 if total > 0 else 0

        topic_results[topic] = {
            "correct_base": correct_base,
            "correct_cot": correct_cot,
            "total": total,
            "accuracy_base": accuracy_base,
            "accuracy_cot": accuracy_cot
        }

    print("\nFinal Results:")
    for topic, results in topic_results.items():
        print(f"{topic}: Base Accuracy = {results['accuracy_base']:.2f}%, CoT Accuracy = {results['accuracy_cot']:.2f}% ({results['correct_base']}/{results['total']} base, {results['correct_cot']}/{results['total']} CoT)")
        # print(f"{topic}: Base Accuracy = {results['accuracy_base']:.2f}%, ({results['correct_base']}/{results['total']} base")
    output_txt_file = "results.txt"

    with open(output_txt_file, "w") as f:
        for topic, results in topic_results.items():
            result_line = (
                f"{topic}: Base Accuracy = {results['accuracy_base']:.2f}%, "
                f"CoT Accuracy = {results['accuracy_cot']:.2f}% "
                f"({results['correct_base']}/{results['total']} base, {results['correct_cot']}/{results['total']} CoT)\n"
            )
            f.write(result_line)
    print(f"Results saved to {output_txt_file}")
    
if __name__ == "__main__":
    folder_path = "MATH/test"  # Path to the training dataset
    model_name = "meta-llama/Llama-3.1-8B-Instruct"  # Pretrained model name (can be replaced with another model)

    base_prompt = """Your goal is to solve a given math problem. 
        Solve a given math problem.
        
        Here are some examples:

        Example 1: Q: Natalia sold paperclips to 48 of her friends in April, and then sold half that number in May.
        How many clips did Natalia sell in total in April and May?
        Answer: ... #### 72

        Example 2: Question: Weng earns $12 per hour as a babysitter. Yesterday, she babysat for 50
        minutes of babysitting. How much did she earn?
        Answer: ... #### 10

        The answer is formatted so that it uses calculation annotations and so that the final numerical solution is forced to place the last line of the solution and preceded by ####.
        
        Output format example(Not real answer):
        (Prev.) ... Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
        """
    


    cot_prompt = """Solve the following math problem step-by-step: {problem}

        Clearly identify intermediate steps and derive the final result.

        Solve a given math problem.
        
        Here are some examples:

        Example 1: Q: Natalia sold paperclips to 48 of her friends in April, and then sold half that number in May.
        How many clips did Natalia sell in total in April and May?
        
        step 1: Number of clips she sold in April: <48>
        step 2: Calculate how many clips Natalia sold in May: <48/2=24>
        step 3: Calculate the total number of sales: <48+24=72>

        Answer: ... #### 72

        Example 2: Question: Weng earns $12 per hour as a babysitter. Yesterday, she babysat for 50
        minutes of babysitting. How much did she earn?

        step 1:  Determine Weng's hourly wage: <12> for one hour
        step 2: Use hourly wage to calculate his pay per minute: <12/60=0.2> per minute
        step 3: Calculate the amount earned in 50 minutes: <0.2*50=10>

        Answer: ... #### 10

        The answer is formatted so that it uses calculation annotations and so that the final numerical solution is forced to place the last line of the solution and preceded by ####.
        
        Output format example(Not real answer):
        (Prev.) ... Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72
        """

    evaluate_dataset_by_topic_with_prompt(folder_path, base_prompt, cot_prompt, model_name, limit=61)
