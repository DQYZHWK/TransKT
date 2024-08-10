
import json
import random


file_path = "new_dataset.jsonl"

with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

random.shuffle(lines)

total_lines = len(lines)
train_size = int(0.7 * total_lines)
dev_size = int(0.2 * total_lines)

train_set = lines[:train_size]
dev_set = lines[train_size:train_size + dev_size]
test_set = lines[train_size + dev_size:]

train_file_path = "train.jsonl"
dev_file_path = "dev.jsonl"
test_file_path = "test.jsonl"

with open(train_file_path, 'w', encoding='utf-8') as train_file:
    train_file.writelines(train_set)

with open(dev_file_path, 'w', encoding='utf-8') as dev_file:
    dev_file.writelines(dev_set)

with open(test_file_path, 'w', encoding='utf-8') as test_file:
    test_file.writelines(test_set)
