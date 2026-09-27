import os
import sys
import re
import unicodedata
from collections import defaultdict, Counter
import time
import numpy as np
import xgboost as xgb
import joblib

sys.stdout.reconfigure(encoding='utf-8')

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")

# Devanagari & Tamil rule-based transliteration maps
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

def clean_record(name, addr, ctry):
    # Name
    clean_n = transliterate(name).lower()
    if DOMAIN_REGEX.search(clean_n):
        clean_n = DOMAIN_REGEX.sub('', clean_n)
    n_toks = re.findall(r'[a-z0-9]+', clean_n)
    n_filt = [t for t in n_toks if t not in LEGAL_STOPWORDS and len(t) > 1]
    n_cat = "".join(n_toks)
    n_3g = {n_cat[i:i+3] for i in range(len(n_cat)-2)} if len(n_cat) >= 3 else ({n_cat} if n_cat else set())
    
    # Address
    clean_a = transliterate(addr).lower()
    raw_digits = re.findall(r'\d+', clean_a)
    digits = [d.lstrip('0') or '0' for d in raw_digits if len(d) <= 10]
    a_toks = re.findall(r'[a-z]+', clean_a)
    states = [t for t in a_toks if t in STATE_CODES]
    words = [t for t in a_toks if t not in ADDR_STOPWORDS and t not in STATE_CODES and len(t) > 2]
    w_cat = "".join(words)
    a_3g = {w_cat[i:i+3] for i in range(len(w_cat)-2)} if len(w_cat) >= 3 else ({w_cat} if w_cat else set())
    
    addr_empty = 1.0 if not clean_a or clean_a in ('null', '<null>', 'none') else 0.0
    
    return {
        'n_filt': n_filt,
        'n_filt_set': set(n_filt),
        'n_cat': n_cat,
        'n_3g': n_3g,
        'digits': digits,
        'digits_set': set(digits),
        'words': words,
        'words_set': set(words),
        'a_3g': a_3g,
        'states': states,
        'addr_empty': addr_empty,
        'ctry': ctry
    }

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

def extract_features(s1, t, b_rank, b_score, n_cands, max_b_score, tid):
    # Name features
    n_jacc = jaccard(s1['n_filt_set'], t['n_filt_set'])
    n_overlap = overlap_coef(s1['n_filt_set'], t['n_filt_set'])
    n_exact = 1.0 if s1['n_cat'] == t['n_cat'] and len(s1['n_cat']) > 0 else 0.0
    n_contains = 1.0 if (s1['n_cat'] in t['n_cat'] or t['n_cat'] in s1['n_cat']) and min(len(s1['n_cat']), len(t['n_cat'])) >= 4 else 0.0
    n_3g_jacc = jaccard(s1['n_3g'], t['n_3g'])
    pref3 = 1.0 if s1['n_cat'][:3] == t['n_cat'][:3] and len(s1['n_cat']) >= 3 else 0.0
    pref6 = 1.0 if s1['n_cat'][:6] == t['n_cat'][:6] and len(s1['n_cat']) >= 6 else 0.0

    # Address features
    t_addr_empty = t['addr_empty']
    d_jacc = jaccard(s1['digits_set'], t['digits_set'])
    d_overlap = overlap_coef(s1['digits_set'], t['digits_set'])
    d_exact = 1.0 if len(s1['digits_set'] & t['digits_set']) > 0 else 0.0

    w_jacc = jaccard(s1['words_set'], t['words_set'])
    w_overlap = overlap_coef(s1['words_set'], t['words_set'])
    a_3g_jacc = jaccard(s1['a_3g'], t['a_3g'])

    # State match
    if s1['states'] and t['states']:
        state_match = 1.0 if set(s1['states']) & set(t['states']) else -1.0
    else:
        state_match = 0.0

    # Cross features
    has_num_and_name = 1.0 if (d_exact > 0 and n_overlap > 0) else 0.0
    has_num_and_word = 1.0 if (d_exact > 0 and w_overlap > 0) else 0.0
    is_s2 = 1.0 if tid.startswith('S2-') else 0.0
    
    score_rel = b_score / max(0.1, max_b_score)

    feats = [
        n_jacc, n_overlap, n_exact, n_contains, n_3g_jacc, pref3, pref6,
        t_addr_empty, d_jacc, d_overlap, d_exact, w_jacc, w_overlap, a_3g_jacc,
        state_match, has_num_and_name, has_num_and_word, is_s2,
        1.0 / (b_rank + 1), b_score, score_rel, 1.0 / (n_cands + 1)
    ]
    return feats

def main():
    print("="*70)
    print("TRAINING PRODUCTION XGBOOST ENTITY RESOLUTION MODEL")
    print("="*70)
    
    # Load 15,000 train S1 entities and 5,000 validation S1 entities
    train_s1 = {}
    val_s1 = {}
    with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
        next(f)
        for i, line in enumerate(f):
            parts = line.rstrip('\n').split('\t')
            s1 = parts[0]
            targets = parts[1].split(',') if len(parts) > 1 and parts[1] else []
            if i < 15000:
                train_s1[s1] = targets
            elif i < 20000:
                val_s1[s1] = targets
            else:
                break
                
    all_needed_s1 = set(train_s1.keys()) | set(val_s1.keys())
    all_needed_targets = {t for ts in train_s1.values() for t in ts} | {t for ts in val_s1.values() for t in ts}
    print(f"Train S1: {len(train_s1)} | Val S1: {len(val_s1)} | Total true target matches: {len(all_needed_targets)}")

    print("Loading S1 records...")
    s1_dict = {}
    with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if parts[0] in all_needed_s1:
                s1_dict[parts[0]] = clean_record(parts[1], parts[2], parts[3])

    print("Loading target records pool (all true targets + 250,000 random distractors)...")
    target_dict = {}
    t_count = 0
    for src in ["train_source2.tsv", "train_source3.tsv"]:
        with open(os.path.join(train_dir, src), 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                t_count += 1
                parts = line.rstrip('\n').split('\t')
                tid = parts[0]
                if tid in all_needed_targets or (t_count % 30 == 0 and len(target_dict) < 280000):
                    target_dict[tid] = clean_record(parts[1], parts[2], parts[3])

    print(f"Total target pool: {len(target_dict)} records")

    print("Building multi-key inverted indexes...")
    t0 = time.time()
    name_index = defaultdict(list)
    name_prefix_index = defaultdict(list)
    name_word0_prefix4 = defaultdict(list)
    addr_num_word_index = defaultdict(list)
    addr_state_digit = defaultdict(list)
    addr_two_num_index = defaultdict(list)

    token_df = Counter()
    prefix4_df = Counter()
    for tid, rec in target_dict.items():
        for t in rec['n_filt_set']:
            token_df[t] += 1
        if rec['n_filt'] and len(rec['n_filt'][0]) >= 4:
            prefix4_df[rec['n_filt'][0][:4]] += 1

    MAX_POSTING = 800
    for tid, rec in target_dict.items():
        ctry = rec['ctry']
        for t in rec['n_filt_set']:
            if token_df[t] <= MAX_POSTING:
                name_index[(ctry, t)].append(tid)
        if len(rec['n_filt']) >= 2:
            name_prefix_index[(ctry, rec['n_filt'][0] + "_" + rec['n_filt'][1])].append(tid)
        elif len(rec['n_filt']) == 1:
            name_prefix_index[(ctry, rec['n_filt'][0])].append(tid)
        if rec['n_filt'] and len(rec['n_filt'][0]) >= 4 and prefix4_df[rec['n_filt'][0][:4]] <= 250:
            name_word0_prefix4[(ctry, rec['n_filt'][0][:4])].append(tid)
        for d in rec['digits'][:2]:
            for w in rec['words'][:3]:
                addr_num_word_index[(ctry, d, w)].append(tid)
        if rec['states'] and rec['digits']:
            for s in rec['states'][:1]:
                for d in rec['digits'][:2]:
                    addr_state_digit[(ctry, s, d)].append(tid)
        if len(rec['digits']) >= 2:
            d1, d2 = sorted(rec['digits'][:3])[:2]
            addr_two_num_index[(ctry, d1, d2)].append(tid)

    print(f"Indexes built in {time.time() - t0:.2f}s")

    def get_candidates(s1, K_MAX=10):
        ctry = s1['ctry']
        cand_scores = defaultdict(float)
        for t in s1['n_filt_set']:
            df = token_df.get(t, 0)
            if 0 < df <= MAX_POSTING:
                w_score = (1.0 / (df ** 0.5)) * 3.5
                for tid in name_index.get((ctry, t), []):
                    cand_scores[tid] += w_score
        if len(s1['n_filt']) >= 2:
            for tid in name_prefix_index.get((ctry, s1['n_filt'][0] + "_" + s1['n_filt'][1]), []):
                cand_scores[tid] += 3.5
        elif len(s1['n_filt']) == 1:
            for tid in name_prefix_index.get((ctry, s1['n_filt'][0]), []):
                cand_scores[tid] += 2.5
        if s1['n_filt'] and len(s1['n_filt'][0]) >= 4 and prefix4_df.get(s1['n_filt'][0][:4], 0) <= 250:
            for tid in name_word0_prefix4.get((ctry, s1['n_filt'][0][:4]), []):
                cand_scores[tid] += 2.0
        for d in s1['digits'][:2]:
            for w in s1['words'][:3]:
                for tid in addr_num_word_index.get((ctry, d, w), []):
                    cand_scores[tid] += 3.0
        if s1['states'] and s1['digits']:
            for s in s1['states'][:1]:
                for d in s1['digits'][:2]:
                    for tid in addr_state_digit.get((ctry, s, d), []):
                        cand_scores[tid] += 2.0
        if len(s1['digits']) >= 2:
            d1, d2 = sorted(s1['digits'][:3])[:2]
            for tid in addr_two_num_index.get((ctry, d1, d2), []):
                cand_scores[tid] += 3.0
        if not cand_scores:
            return []
        sorted_cands = sorted(cand_scores.items(), key=lambda x: x[1], reverse=True)
        return [(tid, score) for tid, score in sorted_cands[:K_MAX] if score >= 1.5]

    print("Generating features for training set (15,000 S1 queries)...")
    t0 = time.time()
    X_train, y_train = [], []
    for s1_id, true_targets in train_s1.items():
        s1 = s1_dict[s1_id]
        true_set = set(true_targets)
        cands = get_candidates(s1, K_MAX=10)
        max_score = cands[0][1] if cands else 1.0
        for rank, (tid, b_score) in enumerate(cands):
            t = target_dict[tid]
            feat = extract_features(s1, t, rank, b_score, len(cands), max_score, tid)
            label = 1 if tid in true_set else 0
            X_train.append(feat)
            y_train.append(label)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"Features generated in {time.time() - t0:.2f}s | Shape: {X_train.shape} | Positives: {np.sum(y_train)} ({np.mean(y_train)*100:.2f}%)")

    # Train XGBoost Classifier
    print("Training production XGBoost model...")
    clf = xgb.XGBClassifier(
        n_estimators=160,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=8,
        tree_method='hist'
    )
    clf.fit(X_train, y_train)

    # Save trained model
    model_dir = os.path.join(base_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgb_matching_model.json")
    clf.save_model(model_path)
    print(f"Model saved to {model_path}")

    # Validation evaluation & threshold tuning
    print("\nRunning Validation Evaluation (5,000 S1 queries)...")
    val_cand_data = []
    for s1_id, true_targets in val_s1.items():
        s1 = s1_dict[s1_id]
        cands = get_candidates(s1, K_MAX=10)
        max_score = cands[0][1] if cands else 1.0
        feats = []
        for rank, (tid, b_score) in enumerate(cands):
            t = target_dict[tid]
            feat = extract_features(s1, t, rank, b_score, len(cands), max_score, tid)
            feats.append((tid, feat))
        val_cand_data.append((s1_id, feats, [t for t, _ in cands]))

    all_val_feats = [f for s1_id, feats, _ in val_cand_data for tid, f in feats]
    all_probs = clf.predict_proba(np.array(all_val_feats, dtype=np.float32))[:, 1]
    
    ptr = 0
    val_prob_map = defaultdict(list)
    val_cand_map = {}
    for s1_id, feats, cand_ids in val_cand_data:
        val_cand_map[s1_id] = cand_ids
        for tid, _ in feats:
            val_prob_map[s1_id].append((tid, all_probs[ptr]))
            ptr += 1

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

    val_gt = {s1: set(ts) for s1, ts in val_s1.items()}
    best_thresh = 0.65
    best_f05 = 0.0

    print("\n--- THRESHOLD TUNING RESULTS ---")
    for thresh in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
        preds = {}
        for s1_id in val_s1.keys():
            cand_probs = val_prob_map.get(s1_id, [])
            preds[s1_id] = {tid for tid, prob in cand_probs if prob >= thresh}
        f05 = compute_macro_f05(val_gt, preds)
        print(f"Threshold: {thresh:.2f} -> Macro F_0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    print(f"\nOPTIMAL VALIDATION F_0.5: {best_f05:.4f} at Threshold: {best_thresh:.2f}")

    # Feature Importance Analysis
    feature_names = [
        "name_jacc", "name_overlap", "name_exact", "name_contains", "name_3g_jacc", "pref3", "pref6",
        "t_addr_empty", "d_jacc", "d_overlap", "d_exact", "w_jacc", "w_overlap", "a_3g_jacc",
        "state_match", "has_num_and_name", "has_num_and_word", "is_s2",
        "b_rank_inv", "b_score", "score_rel", "n_cands_inv"
    ]
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n--- TOP 10 MOST IMPORTANT FEATURES ---")
    for idx in sorted_idx[:10]:
        print(f"{feature_names[idx]:20s}: {importances[idx]:.4f}")

if __name__ == "__main__":
    main()
