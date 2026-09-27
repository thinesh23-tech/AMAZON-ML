import os
import sys
import re
import unicodedata
sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")

def has_non_ascii(s):
    return any(ord(c) > 127 for c in s)

# Let's inspect Indian pairs where target name has non-ascii characters
gt_sample = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) > 1 and parts[1]:
            gt_sample.append((parts[0], parts[1].split(',')))
        if len(gt_sample) >= 3000:
            break

s1_needed = {s1 for s1, _ in gt_sample}
targets_needed = {t for _, ts in gt_sample for t in ts}

s1_data = {}
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[0] in s1_needed and parts[3] == 'India':
            s1_data[parts[0]] = parts

targets_data = {}
for src in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, src), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[0] in targets_needed and parts[3] == 'India':
                targets_data[parts[0]] = parts

print(f"Total Indian S1 entities loaded: {len(s1_data)}")

# For each Indian pair, check:
# 1. Does target name have non-ascii?
# 2. If yes, what is address token overlap?
# 3. What is address digit overlap?
non_ascii_name_count = 0
addr_overlap_count = 0
digit_overlap_count = 0
empty_target_addr_count = 0

for s1, targets in gt_sample:
    s1_row = s1_data.get(s1)
    if not s1_row:
        continue
    s1_addr_toks = set(re.findall(r'\b[a-z0-9]+\b', s1_row[2].lower()))
    s1_digits = {t for t in s1_addr_toks if t.isdigit()}

    for t in targets:
        t_row = targets_data.get(t)
        if not t_row:
            continue
        if has_non_ascii(t_row[1]):
            non_ascii_name_count += 1
            t_addr_toks = set(re.findall(r'\b[a-z0-9]+\b', t_row[2].lower()))
            t_digits = {t for t in t_addr_toks if t.isdigit()}
            if not t_row[2].strip() or t_row[2].lower() in ('null', '<null>'):
                empty_target_addr_count += 1
            if len(s1_addr_toks & t_addr_toks) > 0:
                addr_overlap_count += 1
            if len(s1_digits & t_digits) > 0:
                digit_overlap_count += 1

print(f"Non-ascii target names in India: {non_ascii_name_count}")
print(f"Empty target address among non-ascii: {empty_target_addr_count}")
print(f"Address token overlap among non-ascii: {addr_overlap_count} ({addr_overlap_count/max(1, non_ascii_name_count)*100:.2f}%)")
print(f"Digit overlap among non-ascii: {digit_overlap_count} ({digit_overlap_count/max(1, non_ascii_name_count)*100:.2f}%)")
