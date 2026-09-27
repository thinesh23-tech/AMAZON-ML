import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")

# Read first 100 ground truth rows
gt_pairs = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        s1 = parts[0]
        if len(parts) > 1 and parts[1]:
            matches = parts[1].split(',')
            gt_pairs.append((s1, matches))
        if len(gt_pairs) >= 50:
            break

s1_needed = {s1 for s1, _ in gt_pairs}
targets_needed = {m for _, ms in gt_pairs for m in ms}

s1_data = {}
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[0] in s1_needed:
            s1_data[parts[0]] = parts
        if len(s1_data) == len(s1_needed):
            break

s2_data = {}
with open(os.path.join(train_dir, "train_source2.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[0] in targets_needed:
            s2_data[parts[0]] = parts

s3_data = {}
with open(os.path.join(train_dir, "train_source3.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[0] in targets_needed:
            s3_data[parts[0]] = parts

print("="*80)
print("PRINTING DETAILED POSITIVE MATCH EXAMPLES")
print("="*80)

for s1, ms in gt_pairs[:15]:
    s1_row = s1_data.get(s1)
    if not s1_row:
        continue
    print(f"\n[SOURCE 1] ID: {s1} | Country: {s1_row[3]}")
    print(f"  Name   : {s1_row[1]}")
    print(f"  Address: {s1_row[2]}")
    for m in ms:
        if m in s2_data:
            r = s2_data[m]
            print(f"  --> [S2] ID: {m}")
            print(f"      Name   : {r[1]}")
            print(f"      Address: {r[2]}")
        elif m in s3_data:
            r = s3_data[m]
            print(f"  --> [S3] ID: {m}")
            print(f"      Name   : {r[1]}")
            print(f"      Address: {r[2]}")
