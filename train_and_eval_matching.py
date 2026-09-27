import os
import sys
import re
import unicodedata
from collections import defaultdict, Counter
import time
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb

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
    'unit', 'floor', 'fl', 'box', 'po', 'near', 'behind', 'opp', 'opposite', 'null',
    'township', 'city', 'county', 'state', 'saint', 'north', 'south', 'east', 'west'
}

STATE_CODES = {
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia',
    'ks', 'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj',
    'nm', 'ny', 'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt',
    'va', 'wa', 'wv', 'wi', 'wy', 'dc', 'pr',
    'up', 'mh', 'dl', 'ka', 'tn', 'wb', 'gj', 'rj', 'ap', 'ts', 'kl', 'mp', 'hr', 'pb', 'br', 'or', 'ch'
}

DOMAIN_REGEX = re.compile(r'\.(com|in|org|net|io|co|biz|info|fr|edu|gov)$', re.IGNORECASE)

def clean_name(name):
    name = transliterate(name).lower()
    if DOMAIN_REGEX.search(name):
        name = DOMAIN_REGEX.sub('', name)
    tokens = re.findall(r'[a-z0-9]+', name)
    filtered = [t for t in tokens if t not in LEGAL_STOPWORDS and len(t) > 1]
    return filtered, tokens, "".join(tokens)

def clean_address(addr):
    addr = transliterate(addr).lower()
    raw_digits = re.findall(r'\d+', addr)
    digits = [d.lstrip('0') or '0' for d in raw_digits if len(d) <= 10]
    tokens = re.findall(r'[a-z]+', addr)
    states = [t for t in tokens if t in STATE_CODES]
    words = [t for t in tokens if t not in ADDR_STOPWORDS and t not in STATE_CODES and len(t) > 2]
    return digits, words, states

def get_3grams(s):
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i+3] for i in range(len(s)-2)}

def jaccard(s1, s2):
    if not s1 or not s2:
        return 0.0
    num = len(s1 & s2)
    den = len(s1 | s2)
    return num / den if den > 0 else 0.0

def overlap_coef(s1, s2):
    if not s1 or not s2:
        return 0.0
    num = len(s1 & s2)
    den = min(len(s1), len(s2))
    return num / den if den > 0 else 0.0

def extract_features(s1_tuple, t_tuple, b_rank, b_score, n_cands):
    # s1_tuple: (name_filt, name_tokens, name_concat, digits, words, states, raw_addr)
    # t_tuple: (name_filt, name_tokens, name_concat, digits, words, states, raw_addr)
    s1_n_filt, s1_n_tok, s1_n_cat, s1_dig, s1_w, s1_st, s1_addr = s1_tuple
    t_n_filt, t_n_tok, t_n_cat, t_dig, t_w, t_st, t_addr = t_tuple

    # 1. Name features
    s1_nf_set, t_nf_set = set(s1_n_filt), set(t_n_filt)
    n_jacc = jaccard(s1_nf_set, t_nf_set)
    n_overlap = overlap_coef(s1_nf_set, t_nf_set)
    n_exact = 1.0 if s1_n_cat == t_n_cat and len(s1_n_cat) > 0 else 0.0
    n_contains = 1.0 if (s1_n_cat in t_n_cat or t_n_cat in s1_n_cat) and min(len(s1_n_cat), len(t_n_cat)) >= 4 else 0.0
    
    # 3-gram char jaccard
    s1_3g = get_3grams(s1_n_cat)
    t_3g = get_3grams(t_n_cat)
    n_3g_jacc = jaccard(s1_3g, t_3g)

    # Prefix match
    pref_match3 = 1.0 if s1_n_cat[:3] == t_n_cat[:3] and len(s1_n_cat) >= 3 else 0.0
    pref_match6 = 1.0 if s1_n_cat[:6] == t_n_cat[:6] and len(s1_n_cat) >= 6 else 0.0

    # 2. Address features
    t_addr_empty = 1.0 if not t_addr or t_addr.lower() in ('null', '<null>') else 0.0
    
    s1_dig_set, t_dig_set = set(s1_dig), set(t_dig)
    d_jacc = jaccard(s1_dig_set, t_dig_set)
    d_overlap = overlap_coef(s1_dig_set, t_dig_set)
    d_exact_match = 1.0 if len(s1_dig_set & t_dig_set) > 0 else 0.0

    s1_w_set, t_w_set = set(s1_w), set(t_w)
    w_jacc = jaccard(s1_w_set, t_w_set)
    w_overlap = overlap_coef(s1_w_set, t_w_set)

    # State match
    if s1_st and t_st:
        state_match = 1.0 if set(s1_st) & set(t_st) else -1.0
    else:
        state_match = 0.0

    # Compound name+addr
    has_num_and_name = 1.0 if (d_exact_match > 0 and n_overlap > 0) else 0.0
    has_num_and_word = 1.0 if (d_exact_match > 0 and w_overlap > 0) else 0.0

    feats = [
        n_jacc, n_overlap, n_exact, n_contains, n_3g_jacc, pref_match3, pref_match6,
        t_addr_empty, d_jacc, d_overlap, d_exact_match, w_jacc, w_overlap,
        state_match, has_num_and_name, has_num_and_word,
        1.0 / (b_rank + 1), b_score, 1.0 / (n_cands + 1)
    ]
    return feats

print("Preparing train and validation splits from ground truth...")
# 4,000 for training matcher, 2,000 for held-out validation scoring
train_s1 = {}
val_s1 = {}
with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for i, line in enumerate(f):
        parts = line.rstrip('\n').split('\t')
        s1 = parts[0]
        targets = parts[1].split(',') if len(parts) > 1 and parts[1] else []
        if i < 4000:
            train_s1[s1] = targets
        elif i < 6000:
            val_s1[s1] = targets
        else:
            break

all_train_targets = {t for ts in train_s1.values() for t in ts}
all_val_targets = {t for ts in val_s1.values() for t in ts}
all_needed_s1 = set(train_s1.keys()) | set(val_s1.keys())
all_needed_targets = all_train_targets | all_val_targets

print(f"Train S1: {len(train_s1)} | Val S1: {len(val_s1)} | Total needed true targets: {len(all_needed_targets)}")

# Load S1 records
s1_preprocessed = {}
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        eid = parts[0]
        if eid in all_needed_s1:
            nf, nt, nc = clean_name(parts[1])
            d, w, st = clean_address(parts[2])
            s1_preprocessed[eid] = (nf, nt, nc, d, w, st, parts[2], parts[3])

# Load Target pool (true targets + 150,000 distractors)
target_preprocessed = {}
t_count = 0
for src in ["train_source2.tsv", "train_source3.tsv"]:
    with open(os.path.join(train_dir, src), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            t_count += 1
            parts = line.rstrip('\n').split('\t')
            tid = parts[0]
            if tid in all_needed_targets or (t_count % 35 == 0 and len(target_preprocessed) < 160000):
                nf, nt, nc = clean_name(parts[1])
                d, w, st = clean_address(parts[2])
                target_preprocessed[tid] = (nf, nt, nc, d, w, st, parts[2], parts[3])

print(f"Target pool loaded: {len(target_preprocessed)} records")

# Build inverted indexes for blocking
print("Indexing target pool...")
name_index = defaultdict(list)
name_prefix_index = defaultdict(list)
name_word0_prefix4 = defaultdict(list)
addr_num_word_index = defaultdict(list)
addr_state_digit = defaultdict(list)
addr_two_num_index = defaultdict(list)

token_df = Counter()
prefix4_df = Counter()
for tid, t_data in target_preprocessed.items():
    nf = t_data[0]
    for t in set(nf):
        token_df[t] += 1
    if nf and len(nf[0]) >= 4:
        prefix4_df[nf[0][:4]] += 1

MAX_POSTING = 800
for tid, (nf, nt, nc, digits, words, states, raw_addr, ctry) in target_preprocessed.items():
    for t in set(nf):
        if token_df[t] <= MAX_POSTING:
            name_index[(ctry, t)].append(tid)
    if len(nf) >= 2:
        name_prefix_index[(ctry, nf[0] + "_" + nf[1])].append(tid)
    elif len(nf) == 1:
        name_prefix_index[(ctry, nf[0])].append(tid)
    if nf and len(nf[0]) >= 4 and prefix4_df[nf[0][:4]] <= 250:
        name_word0_prefix4[(ctry, nf[0][:4])].append(tid)
    for d in digits[:2]:
        for w in words[:3]:
            addr_num_word_index[(ctry, d, w)].append(tid)
    if states and digits:
        for s in states[:1]:
            for d in digits[:2]:
                addr_state_digit[(ctry, s, d)].append(tid)
    if len(digits) >= 2:
        d1, d2 = sorted(digits[:3])[:2]
        addr_two_num_index[(ctry, d1, d2)].append(tid)

def get_candidates(s1_data, K_MAX=10):
    nf, nt, nc, digits, words, states, raw_addr, ctry = s1_data
    cand_scores = defaultdict(float)
    for t in set(nf):
        df = token_df.get(t, 0)
        if 0 < df <= MAX_POSTING:
            w_score = (1.0 / (df ** 0.5)) * 3.5
            for tid in name_index.get((ctry, t), []):
                cand_scores[tid] += w_score
    if len(nf) >= 2:
        for tid in name_prefix_index.get((ctry, nf[0] + "_" + nf[1]), []):
            cand_scores[tid] += 3.5
    elif len(nf) == 1:
        for tid in name_prefix_index.get((ctry, nf[0]), []):
            cand_scores[tid] += 2.5
    if nf and len(nf[0]) >= 4 and prefix4_df.get(nf[0][:4], 0) <= 250:
        for tid in name_word0_prefix4.get((ctry, nf[0][:4]), []):
            cand_scores[tid] += 2.0
    for d in digits[:2]:
        for w in words[:3]:
            for tid in addr_num_word_index.get((ctry, d, w), []):
                cand_scores[tid] += 3.0
    if states and digits:
        for s in states[:1]:
            for d in digits[:2]:
                for tid in addr_state_digit.get((ctry, s, d), []):
                    cand_scores[tid] += 2.0
    if len(digits) >= 2:
        d1, d2 = sorted(digits[:3])[:2]
        for tid in addr_two_num_index.get((ctry, d1, d2), []):
            cand_scores[tid] += 3.0
    if not cand_scores:
        return []
    sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
    return [(tid, score) for tid, score in sorted_cands[:K_MAX] if score >= 1.5]

print("Extracting features for training pairs...")
X_train, y_train = [], []
for s1_id, true_targets in train_s1.items():
    s1_data = s1_preprocessed[s1_id]
    true_set = set(true_targets)
    cands = get_candidates(s1_data, K_MAX=10)
    for rank, (tid, b_score) in enumerate(cands):
        t_data = target_preprocessed[tid]
        feat = extract_features(s1_data[:7], t_data[:7], rank, b_score, len(cands))
        label = 1 if tid in true_set else 0
        X_train.append(feat)
        y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
print(f"X_train shape: {X_train.shape} | Positives: {np.sum(y_train)} ({np.mean(y_train)*100:.2f}%)")

# Train XGBoost Model
print("Training XGBoost Classifier...")
clf = xgb.XGBClassifier(
    n_estimators=120,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=4
)
clf.fit(X_train, y_train)

# Validation evaluation & F_0.5 score
print("Evaluating on Validation set...")
def compute_macro_f05(gt_dict, pred_dict):
    scores = []
    for s1_id, true_set in gt_dict.items():
        pred_set = pred_dict.get(s1_id, set())
        if len(true_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
        else:
            if len(pred_set) == 0:
                scores.append(0.0)
            else:
                tp = len(true_set & pred_set)
                if tp == 0:
                    scores.append(0.0)
                else:
                    prec = tp / len(pred_set)
                    rec = tp / len(true_set)
                    scores.append((1.25 * prec * rec) / (0.25 * prec + rec))
    return float(np.mean(scores))

# Generate predictions on validation set for different probability thresholds
val_candidate_data = []
for s1_id, true_targets in val_s1.items():
    s1_data = s1_preprocessed[s1_id]
    cands = get_candidates(s1_data, K_MAX=10)
    feats = []
    for rank, (tid, b_score) in enumerate(cands):
        t_data = target_preprocessed[tid]
        feat = extract_features(s1_data[:7], t_data[:7], rank, b_score, len(cands))
        feats.append((tid, feat))
    val_candidate_data.append((s1_id, feats))

# Batch predict probabilities
all_val_feats = [f for s1_id, feats in val_candidate_data for tid, f in feats]
if all_val_feats:
    all_probs = clf.predict_proba(np.array(all_val_feats, dtype=np.float32))[:, 1]
    ptr = 0
    val_prob_map = defaultdict(list)
    for s1_id, feats in val_candidate_data:
        for tid, _ in feats:
            val_prob_map[s1_id].append((tid, all_probs[ptr]))
            ptr += 1

print("\n--- TUNING THRESHOLD FOR MACRO F_0.5 ---")
val_gt = {s1: set(ts) for s1, ts in val_s1.items()}

best_thresh = 0.5
best_score = 0.0

for thresh in [0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    preds = {}
    for s1_id in val_s1.keys():
        cand_probs = val_prob_map.get(s1_id, [])
        matches = {tid for tid, prob in cand_probs if prob >= thresh}
        preds[s1_id] = matches
    score = compute_macro_f05(val_gt, preds)
    print(f"Threshold: {thresh:.2f} -> Macro F_0.5 Score: {score:.4f}")
    if score > best_score:
        best_score = score
        best_thresh = thresh

print(f"\nBest Threshold: {best_thresh:.2f} with Macro F_0.5: {best_score:.4f}")
