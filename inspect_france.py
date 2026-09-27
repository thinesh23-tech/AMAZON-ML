import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")
test_dir = os.path.join(base_dir, "dataset", "test")

# Sample 10 France records from test_source1, test_source2, test_source3
print("=== SAMPLE FRANCE RECORDS (Test Source 1) ===")
with open(os.path.join(test_dir, "test_source1.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    count = 0
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 4 and parts[3] == 'France':
            print(f"ID: {parts[0]} | Name: {parts[1]} | Addr: {parts[2]}")
            count += 1
            if count >= 5:
                break

print("\n=== SAMPLE FRANCE RECORDS (Test Source 2) ===")
with open(os.path.join(test_dir, "test_source2.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    count = 0
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 4 and parts[3] == 'France':
            print(f"ID: {parts[0]} | Name: {parts[1]} | Addr: {parts[2]}")
            count += 1
            if count >= 5:
                break

print("\n=== SAMPLE FRANCE RECORDS (Test Source 3) ===")
with open(os.path.join(test_dir, "test_source3.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    count = 0
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) >= 4 and parts[3] == 'France':
            print(f"ID: {parts[0]} | Name: {parts[1]} | Addr: {parts[2]}")
            count += 1
            if count >= 5:
                break
