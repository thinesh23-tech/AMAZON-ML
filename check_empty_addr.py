import os
import sys

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")
test_dir = os.path.join(base_dir, "dataset", "test")

def check_empty_addr(filepath):
    total = 0
    empty = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            total += 1
            parts = line.rstrip('\n').split('\t')
            addr = parts[2].strip() if len(parts) > 2 else ""
            if not addr or addr.lower() in ('null', '<null>', 'none', 'n/a'):
                empty += 1
    return total, empty

print("Checking empty addresses...")
for name, p in [
    ("Train S1", os.path.join(train_dir, "train_source1.tsv")),
    ("Train S2", os.path.join(train_dir, "train_source2.tsv")),
    ("Train S3", os.path.join(train_dir, "train_source3.tsv")),
    ("Test S1", os.path.join(test_dir, "test_source1.tsv")),
    ("Test S2", os.path.join(test_dir, "test_source2.tsv")),
    ("Test S3", os.path.join(test_dir, "test_source3.tsv")),
]:
    tot, emp = check_empty_addr(p)
    print(f"{name:10s}: {emp:7d} / {tot:7d} ({emp/tot*100:.2f}%) empty/null addresses")
