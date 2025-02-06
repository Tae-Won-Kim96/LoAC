from datasets import Dataset
import jsonlines

data = []

# JSON 데이터 불러오기
with jsonlines.open("data/train.jsonl", "r") as f:
    for line in f:
            data.append(line)
# 데이터 확인
print(data[0])
print(f'final answer: { data[0]["answer"].split("####")[-1].strip() }, length: { len(data[0]["answer"].split("####")[-1].strip())}')