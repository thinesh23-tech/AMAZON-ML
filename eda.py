import os
import sys

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")
test_dir = os.path.join(base_dir, "dataset", "test")

def line_count(filename):
    count = 0
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in f:
            count += 1
    return count

files = [
    ("train_source1.tsv", os.path.join(train_dir, "train_source1.tsv")),
    ("train_source2.tsv", os.path.join(train_dir, "train_source2.tsv")),
    ("train_source3.tsv", os.path.join(train_dir, "train_source3.tsv")),
    ("train_ground_truth.tsv", os.path.join(train_dir, "train_ground_truth.tsv")),
    ("test_source1.tsv", os.path.join(test_dir, "test_source1.tsv")),
    ("test_source2.tsv", os.path.join(test_dir, "test_source2.tsv")),
    ("test_source3.tsv", os.path.join(test_dir, "test_source3.tsv")),
]

print("=== LINE COUNTS ===")
for name, p in files:
    if os.path.exists(p):
        lc = line_count(p)
        size_mb = os.path.getsize(p) / (1024*1024)
        print(f"{name:25s}: {lc:8d} lines (header included) | {size_mb:6.1f} MB")
    else:
        print(f"{name:25s}: NOT FOUND")

print("\n=== SAMPLE ROWS (train_source1.tsv) ===")
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    for i in range(5):
        print(f.readline().rstrip())

print("\n=== SAMPLE ROWS (train_ground_truth.tsv) ===")
with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    for i in range(10):
        print(f.readline().rstrip())
