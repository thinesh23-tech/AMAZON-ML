import os
import sys
import re
import unicodedata
sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")

def extract_tokens(text):
    if not text:
        return set()
    text = unicodedata.normalize('NFKD', text)
    return set(re.findall(r'\b[a-z0-9]+\b', text.lower()))

LEGAL_WORDS = {'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private', 
               'co', 'company', 'services', 'enterprises', 'solutions', 'holdings', 'group', 'sa', 'sarl', 'sas', 'sasu'}

gt_pairs = []
with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) > 1 and parts[1]:
            gt_pairs.append((parts[0], parts[1].split(',')))
        if len(gt_pairs) >= 500:
            break

s1_needed = {s1 for s1, _ in gt_pairs}
targets_needed = {t for _, ts in gt_pairs for t in ts}

s1_data = {}
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[0] in s1_needed:
            s1_data[parts[0]] = parts
        if len(s1_data) == len(s1_needed):
            break

targets_data = {}
for src in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, src), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[0] in targets_needed:
                targets_data[parts[0]] = parts

print("=== EXAMPLES WHERE NAME OVERLAP IS 0 ===")
count = 0
for s1, targets in gt_pairs:
    s1_row = s1_data.get(s1)
    if not s1_row:
        continue
    s1_name_toks = extract_tokens(s1_row[1]) - LEGAL_WORDS
    for t in targets:
        t_row = targets_data.get(t)
        if not t_row:
            continue
        t_name_toks = extract_tokens(t_row[1]) - LEGAL_WORDS
        if len(s1_name_toks & t_name_toks) == 0:
            count += 1
            print(f"\nExample {count}: S1 {s1} vs {t}")
            print(f"  S1 Name   : {s1_row[1]}")
            print(f"  S1 Addr   : {s1_row[2]}")
            print(f"  T  Name   : {t_row[1]}")
            print(f"  T  Addr   : {t_row[2]}")
            if count >= 8:
                break
    if count >= 8:
        break
