import os
import re
import json
import random
from prompt.prompts import *
from collections import Counter
from macm.executor import Execute_steps
from macm.judge import Judge_statement, Judge_answer, Judge_condition
from macm.thinker import Analysis_conditions, Think_thoughts, Think_Steps
from tqdm import tqdm

def check_condition(question,condition, n):
    """
    Use several Judges to check the statement
    Input:
    conditions, unchecked_conditions, the number of the inspectors (List, Str, int)
    Output:
    True/False (bool)
    """
    for _ in range(n):
        flag = Judge_condition(question,condition).strip()
        print(flag)
        if flag == "False":
            return False
    return True

def check_statement(conditions,statement, n):
    """
    Use several Judges to check the statement
    Input:
    conditions, unchecked_conditions, the number of the inspectors (List, Str, int)
    Output:
    True/False (bool)
    """
    for _ in range(n):
        answer = Judge_statement(conditions,statement)
        print(answer)
        if  "False" in answer or "false" in answer:
            return False
    return True


def check_answer(conditions,statement):
    """
    Use several Judges to check the answer
    Input:
    unchecked_conditions, the number of the inspectors (Str, int)
    Output:
    True/False (bool)
    """
    if_got_answer = Judge_answer(conditions,statement)
    print(if_got_answer)
    if "False" in if_got_answer or "false" in if_got_answer:
        return False
    return True


def check_if_got_answer(conditions,statement,n):
    for _ in range(n):
        flag = check_answer(conditions,statement)
        print(flag)
        if flag == False:
            return False
    return True    

def extract_boxed_answer(answer):
    pattern = r'\\boxed\{(.*)(?=\})'

    matches = re.search(pattern, answer)
    if matches:
        match = matches.group(1)
    else:
        match = "No match found"
    return match

def main(question, times, n):
    """
    Input question and get the final answer from muti-Agent got
    Input:
    quesion, the number of times new conditions are identified, the number of the inspectors  (Str, int, int)
    Output:
    final answer (Str)
    """
    min_voters = 3 # min number of voters
    max_voters = 5 # max number of voters
    possible_answers = []
    try:
        voter_count = 0
        tie = True
        
        # Vote
        while tie or voter_count < min_voters:
            voter_count += 1
            print(f"\n# {voter_count} Thinker is analyzing the question...")
            conditions,objectives = Analysis_conditions(question)
            Initial_condition_numbers = len(conditions) # This line will be used for the $while$ mode
            
            # Think thoughts
            # while len(conditions) - Initial_condition_numbers <= times: 
            for time in range(times): # Try to reduce the LLM queries.
                print(f"\n# {voter_count} Thinker is thinking new thoughts...")
                unchecked_conditions = Think_thoughts(conditions,objectives)
                checked_conditions = []
                for unchecked_condition in unchecked_conditions:
                    print(f"\n# {voter_count} Judge is checking conditions...")
                    if check_statement(conditions,unchecked_condition,n):
                        start = unchecked_condition.find("we can get: ")
                        if start != -1:
                            unchecked_condition = unchecked_condition[start + len("we can get: "):]
                            unchecked_condition = unchecked_condition.split("Reason:")[0]
                        checked_conditions.append(unchecked_condition)
                conditions = conditions + checked_conditions
                if_got_answer = check_if_got_answer(conditions,objectives,1)
                if if_got_answer:
                    break
            print(f"\n# {voter_count} thinker is thinking steps...")
            steps = Think_Steps(conditions,objectives)
            
            print(f"\n# {voter_count} Executor is trying to calculate the answer...")
            final_answer = Execute_steps(conditions,objectives,steps)
            
            # Achieve one potiential answer
            Answer = re.search(r'\\boxed\{(.*)(?=\})', final_answer)  
            if Answer:
                Answer_boxed = Answer.group(1)
            else:
                Answer_boxed = "No match found"
            possible_answers.append(Answer_boxed)
            if voter_count >= min_voters:
                counter = Counter(possible_answers)
                most_votes = counter.most_common(1)[0][1]  
                tie_count = len(list(filter(lambda x: x[1] == most_votes, counter.items())))
                
                tie = tie_count > 1
                print("\nThere is a tie vote. We need to add another voter.")
                if voter_count >= max_voters:
                    print("\nReached maximum voter limit.")
                    break
        most_possible_answer, count = counter.most_common(1)[0]
        print(f"\nThe final answer is {most_possible_answer}")
        return most_possible_answer
    except Exception as e:
        print(f"Error processing file: {e}")

import os
import re
import json
import random
from prompt.prompts import *
from collections import Counter
from macm.executor import Execute_steps
from macm.judge import Judge_statement, Judge_answer, Judge_condition
from macm.thinker import Analysis_conditions, Think_thoughts, Think_Steps
from tqdm import tqdm

def check_condition(question,condition, n):
    """
    Use several Judges to check the statement
    Input:
    conditions, unchecked_conditions, the number of the inspectors (List, Str, int)
    Output:
    True/False (bool)
    """
    for _ in range(n):
        if Judge_condition(question,condition).strip() == "False":
            return False
    return True

def check_statement(conditions,statement, n):
    """
    Use several Judges to check the statement
    Input:
    conditions, unchecked_conditions, the number of the inspectors (List, Str, int)
    Output:
    True/False (bool)
    """
    for _ in range(n):
        answer = Judge_statement(conditions,statement)
        if  "False" in answer or "false" in answer:
            return False
    return True


def check_answer(conditions,statement):
    """
    Use several Judges to check the answer
    Input:
    unchecked_conditions, the number of the inspectors (Str, int)
    Output:
    True/False (bool)
    """
    if_got_answer = Judge_answer(conditions,statement)
    if "False" in if_got_answer or "false" in if_got_answer:
        return False
    return True


def check_if_got_answer(conditions,statement,n):
    for _ in range(n):
        if check_answer(conditions,statement) == False:
            return False
    return True    

def extract_boxed_answer(answer):
    pattern = r'\\boxed\{(.*)(?=\})'

    matches = re.search(pattern, answer)
    if matches:
        match = matches.group(1)
    else:
        match = "No match found"
    return match

def main(question, times, n):
    """
    Input question and get the final answer from muti-Agent got
    Input:
    quesion, the number of times new conditions are identified, the number of the inspectors  (Str, int, int)
    Output:
    final answer (Str)
    """
    min_voters = 3 # min number of voters
    max_voters = 5 # max number of voters
    possible_answers = []
    try:
        voter_count = 0
        tie = True
        
        # Vote
        while tie or voter_count < min_voters:
            voter_count += 1
            print(f"\n# {voter_count} Thinker is analyzing the question...")
            conditions,objectives = Analysis_conditions(question)
            Initial_condition_numbers = len(conditions) # This line will be used for the $while$ mode
            
            # Think thoughts
            # while len(conditions) - Initial_condition_numbers <= times: 
            for time in range(times): # Try to reduce the LLM queries.
                print(f"\n# {voter_count} Thinker is thinking new thoughts...")
                unchecked_conditions = Think_thoughts(conditions,objectives)
                checked_conditions = []
                for unchecked_condition in unchecked_conditions:
                    print(f"\n# {voter_count} Judge is checking conditions...")
                    if check_statement(conditions,unchecked_condition,n):
                        start = unchecked_condition.find("we can get: ")
                        if start != -1:
                            unchecked_condition = unchecked_condition[start + len("we can get: "):]
                            unchecked_condition = unchecked_condition.split("Reason:")[0]
                        checked_conditions.append(unchecked_condition)
                conditions = conditions + checked_conditions
                if_got_answer = check_if_got_answer(conditions,objectives,1)
                if if_got_answer:
                    break
            print(f"\n# {voter_count} thinker is thinking steps...")
            steps = Think_Steps(conditions,objectives)
            print(steps)
            
            print(f"\n# {voter_count} Executor is trying to calculate the answer...")
            final_answer = Execute_steps(conditions,objectives,steps)
            print(final_answer)
            
            # Achieve one potiential answer
            Answer = re.search(r'\\boxed\{(.*)(?=\})', final_answer)  
            if Answer:
                Answer_boxed = Answer.group(1)
            else:
                Answer_boxed = "No match found"
            possible_answers.append(Answer_boxed)
            if voter_count >= min_voters:
                counter = Counter(possible_answers)
                most_votes = counter.most_common(1)[0][1]  
                tie_count = len(list(filter(lambda x: x[1] == most_votes, counter.items())))
                
                tie = tie_count > 1
                print("\nThere is a tie vote. We need to add another voter.")
                if voter_count >= max_voters:
                    print("\nReached maximum voter limit.")
                    break
        most_possible_answer, count = counter.most_common(1)[0]
        print(f"\nThe final answer is {most_possible_answer}")
        return most_possible_answer
    except Exception as e:
        print(f"Error processing file: {e}")


def evaluate_dataset_by_topic(root_folder, times, n, limit=20, output_folder=None):
    """
    root_folder: 예) "MATH/test"
    times, n: main()에 넘겨줄 파라미터
    limit: 최대 몇 개의 문제를 평가할지
    output_folder: 결과 파일을 저장할 폴더. (기본=None이면 root_folder에 저장)
    
    최종적으로 topic(서브디렉토리)별로 "topic_results.txt"를 생성.
    """
    if output_folder is None:
        output_folder = root_folder

    # 1) root_folder(MATH/test) 내의 서브디렉토리를 topic으로 간주
    subdirs = [d for d in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, d))]
    topic_results = {}
    for topic in subdirs:
        total = 0
        correct = 0
        topic_path = os.path.join(root_folder, topic)
        # subdirectory 내의 모든 json 파일 찾기
        json_files = []
        for root, dirs, files in os.walk(topic_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

        # limit개를 랜덤으로 뽑으려면:
        random.shuffle(json_files)
        if limit is not None:
            json_files = json_files[:limit]

        print(f"\n=== Evaluating topic: {topic} ===")
        results = []  # (filename, final_answer) 저장

        # 각 json 파일에 대해 평가
        for file_path in tqdm(json_files, desc=f"Topic: {topic}"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                problem = data.get("problem", "")
                solution = data.get("solution", "")  # 필요하다면 사용

                if not problem:
                    continue
                else:
                    total += 1

                # main() 함수를 통해 최종 정답 얻기
                final_answer = main(problem, times, n)
                
                # filename: output 형태로 저장하기 위해
                filename_only = os.path.basename(file_path)
                results.append((filename_only, final_answer))
            
            except Exception as e:
                print(f"Error on file {file_path}: {e}")
            target = extract_boxed_answer(solution)
            if target == final_answer:
                correct += 1

        accuracy = (correct / total) * 100 if total > 0 else 0

        topic_results[topic] = {
            "total": total,
            "correct": correct,
            "accuracy": accuracy
        }
        print(f"{topic}: Base Accuracy = {accuracy:.2f}%, {correct}/{total} MACM")

        # 토픽별 결과를 파일로 저장
        output_file_path = os.path.join(output_folder, f"{topic}_results.txt")
        with open(output_file_path, "w", encoding="utf-8") as out_f:
            for filename, answer in results:
                # filename: answer
                out_f.write(f"{filename}: {answer}\n")

        print(f"Topic '{topic}' results saved to {output_file_path}")
    print("\nFinal Results:")
    for topic, results in topic_results.items():
        print(f"{topic}: Base Accuracy = {results['accuracy']:.2f}%, {results['correct']}/{results['total']} MACM)")
    
    output_txt_file = "results.txt"
    with open(output_txt_file, "w") as f:
        for topic, results in topic_results.items():
            result_line = (
                f"{topic}: Base Accuracy = {results['accuracy']:.2f}%, "
                f"({results['correct']}/{results['total']} MACM\n"
            )
            f.write(result_line)
    print(f"Results saved to {output_txt_file}")
# --------------------------------
# 실제 실행 부분
# --------------------------------
if __name__ == "__main__":
    n = 1  # judge 검증 횟수
    times = 3  # Think_thoughts 반복 횟수

    root_folder = "MATH/test"  # MATH/test 아래에 algebra, geometry 등 토픽 폴더가 존재한다고 가정
    evaluate_dataset_by_topic(
        root_folder=root_folder,
        times=times,
        n=n,
        limit=11,  # 각 토픽당 최대 10문제만 테스트
        output_folder='MATH/test/macm'  # None이면 MATH/test 폴더 안에 결과 파일 저장
    )


# def evaluate_dataset(folder_path, times, n, limit=5):
#     all_files = []
#     for root, dirs, files in os.walk(folder_path):
#         print(root, dirs, files)
#         for file in files:
#             if file.endswith('.json'):
#                 file_path = os.path.join(root, file)
#                 all_files.append(file_path)

#     random.shuffle(all_files)  # Shuffle the order of files randomly

#     for count, file_path in tqdm(enumerate(all_files[:limit])):
#         print(file_path)
#         with open(file_path, 'r') as json_file:
#             try:
#                 data = json.load(json_file)
#                 problem = data.get("problem")
#                 if problem:
#                     print(f"#{count} Problem:\n", problem)
#                     solution = data.get("solution")
#                     print(f"#{count} Solution\n", solution)
#                     main(problem, times, n)
#             except json.JSONDecodeError:
#                 print(f"Error reading file {file_path}")
#             except Exception as e:
#                 print(f"Error processing file {file_path}: {e}")
                            
                                          
# if __name__ == "__main__":
#     n = 1 # verification times
#     times = 3 # The upper limit of the mining times

#     folder_path = "MATH/test"
#     evaluate_dataset(folder_path, times, n, limit=29)
