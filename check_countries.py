import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")

# Check country distribution in train_source1, 2, 3
from collections import Counter

def get_country_counts(filepath):
    counts = Counter()
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) >= 4:
                counts[parts[3]] += 1
    return counts

print("Train S1 countries:", get_country_counts(os.path.join(train_dir, "train_source1.tsv")))
print("Train S2 countries:", get_country_counts(os.path.join(train_dir, "train_source2.tsv")))
print("Train S3 countries:", get_country_counts(os.path.join(train_dir, "train_source3.tsv")))

test_dir = os.path.join(base_dir, "dataset", "test")
print("Test S1 countries:", get_country_counts(os.path.join(test_dir, "test_source1.tsv")))
print("Test S2 countries:", get_country_counts(os.path.join(test_dir, "test_source2.tsv")))
print("Test S3 countries:", get_country_counts(os.path.join(test_dir, "test_source3.tsv")))
