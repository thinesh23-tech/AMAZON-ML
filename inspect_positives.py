import os

base_dir = r"c:\Users\Thinesh\Downloads\6ab10eb3b23ba_student_resource\student_resource"
train_dir = os.path.join(base_dir, "dataset", "train")

# Read first 100 non-empty ground truth rows
gt_sample = {}
singletons = 0
total_gt = 0
match_counts = []

with open(os.path.join(train_dir, "train_ground_truth.tsv"), 'r', encoding='utf-8') as f:
    header = next(f)
    for line in f:
        total_gt += 1
        parts = line.rstrip('\n').split('\t')
        s1 = parts[0]
        matches = parts[1].split(',') if len(parts) > 1 and parts[1] else []
        match_counts.append(len(matches))
        if len(matches) == 0:
            singletons += 1
        elif len(gt_sample) < 20:
            gt_sample[s1] = matches
        if total_gt % 500000 == 0:
            print(f"Processed {total_gt} GT rows...")

print(f"Total S1 in train: {total_gt}")
print(f"Singletons: {singletons} ({singletons/total_gt*100:.2f}%)")
print(f"Average matches per S1: {sum(match_counts)/len(match_counts):.2f}")
print(f"Max matches: {max(match_counts)}")

# Let's inspect the actual records for the first 5 S1 entities in gt_sample
sample_s1_ids = set(list(gt_sample.keys())[:5])
sample_target_ids = set()
for s1 in sample_s1_ids:
    for m in gt_sample[s1]:
        sample_target_ids.add(m)

s1_records = {}
with open(os.path.join(train_dir, "train_source1.tsv"), 'r', encoding='utf-8') as f:
    header = next(f)
    for line in f:
        eid, name, addr, ctry = line.rstrip('\n').split('\t')
        if eid in sample_s1_ids:
            s1_records[eid] = (name, addr, ctry)
        if len(s1_records) == len(sample_s1_ids):
            break

s2_records = {}
with open(os.path.join(train_dir, "train_source2.tsv"), 'r', encoding='utf-8') as f:
    header = next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[0] in sample_target_ids:
            s2_records[parts[0]] = (parts[1], parts[2], parts[3])

s3_records = {}
with open(os.path.join(train_dir, "train_source3.tsv"), 'r', encoding='utf-8') as f:
    header = next(f)
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if parts[0] in sample_target_ids:
            s3_records[parts[0]] = (parts[1], parts[2], parts[3])

print("\n" + "="*80)
print("INSPECTING POSITIVE MATCH EXAMPLES")
print("="*80)
for s1_id in sample_s1_ids:
    s1_name, s1_addr, s1_ctry = s1_records[s1_id]
    print(f"\n[SOURCE 1] ID: {s1_id} | Country: {s1_ctry}")
    print(f"  Name   : {s1_name}")
    print(f"  Address: {s1_addr}")
    print("  --- Matches ---")
    for m in gt_sample[s1_id]:
        if m in s2_records:
            m_name, m_addr, m_ctry = s2_records[m]
            print(f"  [S2] ID: {m} | Country: {m_ctry}")
            print(f"    Name   : {m_name}")
            print(f"    Address: {m_addr}")
        elif m in s3_records:
            m_name, m_addr, m_ctry = s3_records[m]
            print(f"  [S3] ID: {m} | Country: {m_ctry}")
            print(f"    Name   : {m_name}")
            print(f"    Address: {m_addr}")
