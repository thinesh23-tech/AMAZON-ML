import os
import sys
import re
import unicodedata
sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")

# Let's inspect 100 positive pairs from India and 100 from US
# Specifically looking at what tokens they share between S1 and S2/S3
gt_india = []
gt_us = []

# Load first 2000 GT
with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) > 1 and parts[1]:
            s1 = parts[0]
            targets = parts[1].split(',')
            # We don't know country yet, collect
            gt_india.append((s1, targets))
        if len(gt_india) >= 2000:
            break

s1_needed = {s1 for s1, _ in gt_india}
targets_needed = {t for _, ts in gt_india for t in ts}

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

# Analyze token overlap in Name and Address
def extract_tokens(text):
    if not text:
        return set()
    # Normalize unicode
    text = unicodedata.normalize('NFKD', text)
    # Extract alphanumeric words lowercased
    words = re.findall(r'\b[a-z0-9]+\b', text.lower())
    return set(words)

# Common legal noise words to ignore
LEGAL_WORDS = {'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private', 
               'co', 'company', 'services', 'enterprises', 'solutions', 'holdings', 'group', 'sa', 'sarl', 'sas', 'sasu'}

name_overlaps = []
addr_overlaps = []
combined_overlaps = []
digit_overlaps = []

for s1, targets in gt_india:
    s1_row = s1_data.get(s1)
    if not s1_row:
        continue
    s1_name_toks = extract_tokens(s1_row[1]) - LEGAL_WORDS
    s1_addr_toks = extract_tokens(s1_row[2])
    s1_digits = {t for t in s1_addr_toks if t.isdigit()}
    s1_all = s1_name_toks | s1_addr_toks

    for t in targets:
        t_row = s2_data.get(t) or s3_data.get(t)
        if not t_row:
            continue
        t_name_toks = extract_tokens(t_row[1]) - LEGAL_WORDS
        t_addr_toks = extract_tokens(t_row[2])
        t_digits = {w for w in t_addr_toks if w.isdigit()}
        t_all = t_name_toks | t_addr_toks

        name_int = len(s1_name_toks & t_name_toks)
        addr_int = len(s1_addr_toks & t_addr_toks)
        comb_int = len(s1_all & t_all)
        dig_int = len(s1_digits & t_digits)

        name_overlaps.append(name_int > 0)
        addr_overlaps.append(addr_int > 0)
        combined_overlaps.append(comb_int > 0)
        digit_overlaps.append(dig_int > 0 if s1_digits and t_digits else False)

print(f"Total positive pairs analyzed: {len(name_overlaps)}")
print(f"Name token overlap > 0: {sum(name_overlaps)/len(name_overlaps)*100:.2f}%")
print(f"Address token overlap > 0: {sum(addr_overlaps)/len(addr_overlaps)*100:.2f}%")
print(f"Combined token overlap > 0: {sum(combined_overlaps)/len(combined_overlaps)*100:.2f}%")
print(f"Digit overlap > 0 (when both have digits): {sum(digit_overlaps)/len(digit_overlaps)*100:.2f}%")
