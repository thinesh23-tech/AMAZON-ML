import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")

# Only store India IDs since all others are US
print("Loading India IDs...")
s1_india = set()
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[3] == 'India':
            s1_india.add(parts[0])

s2_india = set()
with open(os.path.join(train_dir, "train_source2.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[3] == 'India':
            s2_india.add(parts[0])

s3_india = set()
with open(os.path.join(train_dir, "train_source3.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[3] == 'India':
            s3_india.add(parts[0])

print(f"India counts: S1={len(s1_india)}, S2={len(s2_india)}, S3={len(s3_india)}")

print("Checking ground truth cross-country matches...")
cross_country_matches = 0
total_checked = 0

with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        s1 = parts[0]
        if len(parts) > 1 and parts[1]:
            s1_is_india = s1 in s1_india
            for target in parts[1].split(','):
                total_checked += 1
                if target.startswith("S2-"):
                    target_is_india = target in s2_india
                elif target.startswith("S3-"):
                    target_is_india = target in s3_india
                else:
                    target_is_india = None
                if target_is_india != s1_is_india:
                    cross_country_matches += 1
                    if cross_country_matches <= 5:
                        print(f"Cross country: S1 {s1} (India={s1_is_india}) -> {target} (India={target_is_india})")

print(f"Total checked: {total_checked}")
print(f"Cross country matches: {cross_country_matches}")

