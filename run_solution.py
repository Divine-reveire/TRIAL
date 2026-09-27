"""
Amazon ML Challenge 2026: Ultra-Fast Precision Entity Resolution Pipeline
Target: Macro F_0.5 > 0.99 | Execution Time: < 90s on full million-row dataset
- Uses O(1) hash-based inverted index on normalized tokens & postal codes
- Auto-detects normalized/raw directory structures and headers
- Generates: output/matching_results.tsv and output/candidate_pairs.tsv
"""

import os
import sys
import time
import re
from collections import defaultdict

t_start = time.time()

# --- 1. Find Data Directories ---
def find_dirs():
    root = os.path.dirname(os.path.abspath(__file__))
    current = root
    search_dirs = [current]
    for _ in range(3):
        current = os.path.dirname(current)
        search_dirs.append(current)

    train_dir, test_dir = None, None
    for d in search_dirs:
        for r, dirs, files in os.walk(d):
            # Prefer 'normalised' or 'normalized' paths if present
            if any("train_source1" in f for f in files) or "train_ground_truth.tsv" in files:
                if not train_dir or "normal" in r.lower():
                    train_dir = r
            if any("test_source1" in f for f in files):
                if not test_dir or "normal" in r.lower():
                    test_dir = r

    return train_dir, test_dir

TRAIN_DIR, TEST_DIR = find_dirs()
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(f"[*] Train Dir: {TRAIN_DIR}")
print(f"[*] Test Dir : {TEST_DIR}")
print(f"[*] Output   : {OUTPUT_DIR}")
print("=" * 60)

# --- 2. High-Speed File Parser ---
def parse_tsv_fast(file_path):
    records = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = [c.strip().lower() for c in f.readline().rstrip('\r\n').split('\t')]
        cols = {c: i for i, c in enumerate(header)}
        
        eid_idx = cols.get('entity_id', 0)
        country_idx = cols.get('country', 3 if len(header) > 3 else -1)
        name_idx = cols.get('core_business_name', cols.get('norm_business_name', cols.get('business_name', 1)))
        sorted_idx = cols.get('name_tokens_sorted', -1)
        first_idx = cols.get('name_first_token', -1)
        postal_idx = cols.get('postal_code', -1)
        addr_idx = cols.get('norm_business_address', cols.get('business_address', 2 if len(header) > 2 else -1))
        
        for line in f:
            parts = line.rstrip('\r\n').split('\t')
            if len(parts) <= eid_idx:
                continue
            
            eid = parts[eid_idx]
            country = parts[country_idx].strip().upper() if country_idx >= 0 and len(parts) > country_idx else ''
            name = parts[name_idx].strip().lower() if name_idx >= 0 and len(parts) > name_idx else ''
            
            sorted_name = parts[sorted_idx].strip().lower() if sorted_idx >= 0 and len(parts) > sorted_idx else ''
            if not sorted_name and name:
                sorted_name = ' '.join(sorted(re.findall(r'[\w]+', name, re.UNICODE)))
                
            first_tok = parts[first_idx].strip().lower() if first_idx >= 0 and len(parts) > first_idx else ''
            if not first_tok and sorted_name:
                first_tok = sorted_name.split()[0]
                
            postal = parts[postal_idx].strip() if postal_idx >= 0 and len(parts) > postal_idx else ''
            addr = parts[addr_idx].strip().lower() if addr_idx >= 0 and len(parts) > addr_idx else ''
            
            nums = set(re.findall(r'\b\d{2,6}\b', addr))
            tokens = set(re.findall(r'[\w]+', name, re.UNICODE))
            
            records.append((eid, country, sorted_name, first_tok, postal, nums, tokens))
    return records

# --- 3. Instant Resolution Engine ---
def resolve_entities(s1_records, target_records, max_candidates=15):
    # Inverted indices
    tgt_map = {}
    tok_index = defaultdict(lambda: defaultdict(list))
    sort_index = defaultdict(lambda: defaultdict(list))
    postal_index = defaultdict(lambda: defaultdict(list))

    for tgt in target_records:
        eid, country, sorted_name, first_tok, postal, nums, tokens = tgt
        tgt_map[eid] = tgt
        if first_tok:
            tok_index[country][first_tok].append(eid)
        if sorted_name:
            sort_index[country][sorted_name].append(eid)
        if postal:
            postal_index[country][postal].append(eid)

    candidates = {}
    matches = {}

    for s1 in s1_records:
        eid, country, sorted_name, first_tok, postal, nums, tokens = s1
        cand_ids = set()
        
        if first_tok:
            cand_ids.update(tok_index[country].get(first_tok, []))
        if sorted_name:
            cand_ids.update(sort_index[country].get(sorted_name, []))
        if postal:
            cand_ids.update(postal_index[country].get(postal, []))
            
        c_list = list(cand_ids)[:max_candidates]
        candidates[eid] = c_list

        m_list = []
        for tid in c_list:
            _, _, t_sorted, _, _, t_nums, t_tokens = tgt_map[tid]

            # Conflict prune: if both have street numbers, they must agree
            if nums and t_nums and not (nums & t_nums):
                continue

            # Exact sorted token match
            if sorted_name and sorted_name == t_sorted:
                m_list.append(tid)
                continue

            # Token overlap Jaccard
            if tokens and t_tokens:
                jacc = len(tokens & t_tokens) / len(tokens | t_tokens)
                if jacc >= 0.80:
                    m_list.append(tid)

        matches[eid] = m_list

    return candidates, matches

# --- 4. Main Execution ---
if __name__ == "__main__":
    if not TEST_DIR:
        print("[ERROR] Test directory not found.")
        sys.exit(1)

    s1_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if "source1" in f and f.endswith(".tsv")]
    s2_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if "source2" in f and f.endswith(".tsv")]
    s3_files = [os.path.join(TEST_DIR, f) for f in os.listdir(TEST_DIR) if "source3" in f and f.endswith(".tsv")]

    if not (s1_files and s2_files and s3_files):
        print("[ERROR] Missing test source files in test directory.")
        sys.exit(1)

    print("\nLoading test files...")
    t0 = time.time()
    s1_data = parse_tsv_fast(s1_files[0])
    s2_data = parse_tsv_fast(s2_files[0])
    s3_data = parse_tsv_fast(s3_files[0])
    print(f"Loaded {len(s1_data):,} Source 1, {len(s2_data):,} Source 2, {len(s3_data):,} Source 3 records in {time.time() - t0:.2f}s")

    print("\nRunning resolution...")
    t1 = time.time()
    candidates, matches = resolve_entities(s1_data, s2_data + s3_data)
    print(f"Resolution completed in {time.time() - t1:.2f}s")

    cand_out = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    match_out = os.path.join(OUTPUT_DIR, "matching_results.tsv")

    print("\nWriting output TSVs...")
    with open(cand_out, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s in s1_data:
            eid = s[0]
            f.write(f"{eid}\t{','.join(candidates.get(eid, []))}\n")

    with open(match_out, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1_data:
            eid = s[0]
            f.write(f"{eid}\t{','.join(matches.get(eid, []))}\n")

    print("=" * 60)
    print(f"[SUCCESS] Total run time: {time.time() - t_start:.2f} seconds")
    print(f"Output 1: {match_out}")
    print(f"Output 2: {cand_out}")
    print("=" * 60)