import os
import sys
import re
import unicodedata
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")

DEVA_MAP = {
    0x0905: 'a', 0x0906: 'aa', 0x0907: 'i', 0x0908: 'ee', 0x0909: 'u', 0x090a: 'oo',
    0x090e: 'e', 0x090f: 'e', 0x0910: 'ai', 0x0912: 'o', 0x0913: 'o', 0x0914: 'au',
    0x0915: 'k', 0x0916: 'kh', 0x0917: 'g', 0x0918: 'gh', 0x0919: 'n',
    0x091a: 'ch', 0x091b: 'chh', 0x091c: 'j', 0x091d: 'jh', 0x091e: 'n',
    0x091f: 't', 0x0920: 'th', 0x0921: 'd', 0x0922: 'dh', 0x0923: 'n',
    0x0924: 't', 0x0925: 'th', 0x0926: 'd', 0x0927: 'dh', 0x0928: 'n',
    0x092a: 'p', 0x092b: 'ph', 0x092c: 'b', 0x092d: 'bh', 0x092e: 'm',
    0x092f: 'y', 0x0930: 'r', 0x0932: 'l', 0x0933: 'l', 0x0935: 'v',
    0x0936: 'sh', 0x0937: 'sh', 0x0938: 's', 0x0939: 'h',
    0x093e: 'a', 0x093f: 'i', 0x0940: 'ee', 0x0941: 'u', 0x0942: 'oo',
    0x0947: 'e', 0x0948: 'ai', 0x094b: 'o', 0x094c: 'au',
    0x094d: '', 0x0902: 'n', 0x0901: 'n', 0x0903: 'h',
}
TAMIL_MAP = {
    0x0B85: 'a', 0x0B86: 'aa', 0x0B87: 'i', 0x0B88: 'ee', 0x0B89: 'u', 0x0B8A: 'oo',
    0x0B8E: 'e', 0x0B8F: 'e', 0x0B90: 'ai', 0x0B92: 'o', 0x0B93: 'o', 0x0B94: 'au',
    0x0B95: 'k', 0x0B99: 'n', 0x0B9A: 'ch', 0x0B9C: 'j', 0x0B9E: 'n',
    0x0B9F: 't', 0x0BA3: 'n', 0x0BA4: 't', 0x0BA8: 'n', 0x0BA9: 'n',
    0x0BAA: 'p', 0x0BAE: 'm', 0x0BAF: 'y', 0x0BB0: 'r', 0x0BB1: 'r',
    0x0BB2: 'l', 0x0BB3: 'l', 0x0BB4: 'zh', 0x0BB5: 'v', 0x0BB7: 'sh',
    0x0BB8: 's', 0x0BB9: 'h', 0x0BBE: 'a', 0x0BBF: 'i', 0x0BC0: 'ee', 0x0BC1: 'u',
    0x0BC2: 'oo', 0x0BC6: 'e', 0x0BC7: 'e', 0x0BC8: 'ai', 0x0BCA: 'o', 0x0BCB: 'o',
    0x0BCC: 'au', 0x0BCD: '',
}

def transliterate(text):
    if not any(ord(c) > 127 for c in text):
        return text
    text = unicodedata.normalize('NFKD', text)
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in DEVA_MAP:
            out.append(DEVA_MAP[cp])
        elif cp in TAMIL_MAP:
            out.append(TAMIL_MAP[cp])
        elif 0x0C00 <= cp <= 0x0C7F:
            out.append(DEVA_MAP.get(cp - 0x0C00 + 0x0900, ''))
        elif 0x0C80 <= cp <= 0x0CFF:
            out.append(DEVA_MAP.get(cp - 0x0C80 + 0x0900, ''))
        elif 0x0980 <= cp <= 0x09FF:
            out.append(DEVA_MAP.get(cp - 0x0980 + 0x0900, ''))
        elif cp < 128:
            out.append(ch)
    return "".join(out)

LEGAL_STOPWORDS = {
    'inc', 'incorporated', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'pvt', 'private',
    'co', 'company', 'services', 'service', 'enterprises', 'enterprise', 'solutions', 'solution',
    'holdings', 'holding', 'group', 'sa', 'sarl', 'sas', 'sasu', 'eurl', 'sci', 'snc', 'llp',
    'the', 'and', 'of', 'in', 'at', 'to', 'for', 'a', 'an'
}

ADDR_STOPWORDS = {
    'street', 'st', 'road', 'rd', 'drive', 'dr', 'avenue', 'ave', 'boulevard', 'blvd',
    'lane', 'ln', 'place', 'pl', 'court', 'ct', 'suite', 'ste', 'apartment', 'apt',
    'unit', 'floor', 'fl', 'box', 'po', 'near', 'behind', 'opp', 'opposite', 'null'
}

def clean_name(name):
    name = transliterate(name).lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', name)
    filtered = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]
    return filtered, tokens

def clean_address(addr):
    addr = transliterate(addr).lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', addr)
    digits = [t for t in tokens if t.isdigit() and len(t) <= 8]
    words = [t for t in tokens if not t.isdigit() and t not in ADDR_STOPWORDS and len(t) > 2]
    return digits, words

# Sample 1,000 S1 records
val_s1 = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        val_s1[parts[0]] = parts[1].split(',') if len(parts) > 1 and parts[1] else []
        if len(val_s1) >= 1000:
            break

all_val_targets = {t for ts in val_s1.values() for t in ts}

s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[0] in val_s1:
            s1_records[parts[0]] = (parts[1], parts[2], parts[3])
        if len(s1_records) == len(val_s1):
            break

target_records = {}
for src in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, src), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            tid = parts[0]
            if tid in all_val_targets:
                target_records[tid] = (parts[1], parts[2], parts[3])

# Run existing candidate generation and inspect missed
name_index = defaultdict(list)
addr_num_word_index = defaultdict(list)
name_prefix_index = defaultdict(list)
token_df = Counter()
for tid, (tname, taddr, tctry) in target_records.items():
    n_filt, n_all = clean_name(tname)
    for t in set(n_filt):
        token_df[t] += 1

MAX_POSTING = 500
for tid, (tname, taddr, tctry) in target_records.items():
    n_filt, n_all = clean_name(tname)
    for t in set(n_filt):
        if token_df[t] <= MAX_POSTING:
            name_index[(tctry, t)].append(tid)
    if len(n_filt) >= 2:
        name_prefix_index[(tctry, n_filt[0] + "_" + n_filt[1])].append(tid)
    elif len(n_filt) == 1:
        name_prefix_index[(tctry, n_filt[0])].append(tid)
    digits, words = clean_address(taddr)
    for d in digits[:2]:
        for w in words[:3]:
            addr_num_word_index[(tctry, d, w)].append(tid)

K_MAX = 8
missed_count = 0
print("=== MISSED TRUE MATCHES INSPECTION ===")
for s1_id, (s1_name, s1_addr, s1_ctry) in s1_records.items():
    true_matches = [t for t in val_s1[s1_id] if t in target_records]
    s1_n_filt, s1_n_all = clean_name(s1_name)
    s1_digits, s1_words = clean_address(s1_addr)
    
    cand_scores = defaultdict(float)
    for t in set(s1_n_filt):
        df = token_df.get(t, 0)
        if 0 < df <= MAX_POSTING:
            w_score = (1.0 / (df ** 0.5)) * 2.0
            for tid in name_index.get((s1_ctry, t), []):
                cand_scores[tid] += w_score
                
    if len(s1_n_filt) >= 2:
        for tid in name_prefix_index.get((s1_ctry, s1_n_filt[0] + "_" + s1_n_filt[1]), []):
            cand_scores[tid] += 3.0
    elif len(s1_n_filt) == 1:
        for tid in name_prefix_index.get((s1_ctry, s1_n_filt[0]), []):
            cand_scores[tid] += 2.0
            
    for d in s1_digits[:2]:
        for w in s1_words[:3]:
            for tid in addr_num_word_index.get((s1_ctry, d, w), []):
                cand_scores[tid] += 2.5

    sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
    top_cands = set([tid for tid, score in sorted_cands[:K_MAX] if score >= 1.0])
    
    for tm in true_matches:
        if tm not in top_cands:
            missed_count += 1
            t_name, t_addr, t_ctry = target_records[tm]
            print(f"\nMissed {missed_count}: S1 {s1_id} vs {tm} ({s1_ctry})")
            print(f"  S1 Name: {s1_name} -> tokens: {s1_n_filt}")
            print(f"  S1 Addr: {s1_addr} -> digits: {s1_digits}, words: {s1_words[:4]}")
            tn_filt, _ = clean_name(t_name)
            td_digits, td_words = clean_address(t_addr)
            print(f"  T  Name: {t_name} -> tokens: {tn_filt}")
            print(f"  T  Addr: {t_addr} -> digits: {td_digits}, words: {td_words[:4]}")
            if missed_count >= 8:
                break
    if missed_count >= 8:
        break
