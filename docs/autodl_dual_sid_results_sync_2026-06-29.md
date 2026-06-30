(base) root@autodl-container-a85511aafa-ef91e78f:~/autodl-tmp/projects/MiniOneRec# cd /root/autodl-tmp/projects/MiniOneRec

python - <<'PY'
import json
import pandas as pd
from pathlib import Path

CATS = ["Industrial_and_Scientific", "Office_Products"]
TEXT_VERSION = "text_mbk_k512_dedup"
BEHAVIOR_VERSION = "cf_k512_dedup"

def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def first_jsonl(path, n=3):
    path = Path(path)
    rows = []
    if not path.exists():
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rows.append(json.loads(line))
    return rows

def count_jsonl(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

def find_candidate_file(cat, stream, mode):
    # stream: text or cf
    patterns = [
        f"results/candidates_{cat}_{stream}_k512_30k_{mode}_c500_vdual/candidates.jsonl",
        f"results/candidates_{cat}_{stream}_k512_30k_{mode}_c500*/candidates.jsonl",
    ]
    import glob
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return Path(hits[0])
    return None

def compare_test_csv(cat):
    print("\n" + "=" * 100)
    print(f"[TEST CSV ALIGNMENT] {cat}")
    text_csv = Path(f"data/Amazon/sid_versions/{TEXT_VERSION}/{cat}/test.csv")
    cf_csv = Path(f"data/Amazon/sid_versions/{BEHAVIOR_VERSION}/{cat}/test.csv")

    print("text_csv:", text_csv, "exists=", text_csv.exists())
    print("cf_csv:  ", cf_csv, "exists=", cf_csv.exists())

    if not text_csv.exists() or not cf_csv.exists():
        print("MISSING test.csv, cannot compare.")
        return

    dt = pd.read_csv(text_csv)
PY  inspect_dual_reports(cat)view:", json.dumps(rerank, ensure_ascii=False)[:1000])k, dict) else type(rerank))"])

====================================================================================================
[TEST CSV ALIGNMENT] Industrial_and_Scientific
text_csv: data/Amazon/sid_versions/text_mbk_k512_dedup/Industrial_and_Scientific/test.csv exists= True
cf_csv:   data/Amazon/sid_versions/cf_k512_dedup/Industrial_and_Scientific/test.csv exists= True
len(text), len(cf): 4533 4533
columns(text): ['user_id', 'history_item_title', 'item_title', 'history_item_id', 'item_id', 'history_item_sid', 'item_sid', 'history_item_sid_old', 'item_sid_old']
columns(cf):   ['user_id', 'history_item_title', 'item_title', 'history_item_id', 'item_id', 'history_item_sid', 'item_sid', 'history_item_sid_old', 'item_sid_old']
common identity cols: ['user_id', 'history_item_id', 'item_id']
PASS: row count equal
user_id: PASS equal sequence
history_item_id: PASS equal sequence
item_id: PASS equal sequence

====================================================================================================
[CANDIDATE FILE ALIGNMENT] Industrial_and_Scientific

--- mode: exact ---
text candidates: results/candidates_Industrial_and_Scientific_text_k512_30k_exact_c500_v2/candidates.jsonl exists= True
cf candidates:   results/candidates_Industrial_and_Scientific_cf_k512_30k_exact_c500_v1/candidates.jsonl exists= True
jsonl rows text/cf: 4533 4533 PASS
text first-row keys: ['candidate_details', 'candidate_hit_rank_0_based', 'candidate_item_ids', 'candidate_pool_hit', 'generation_scores', 'history_item_id', 'history_item_sid', 'latency_ms', 'mapped_target_sid', 'pred_sid_valid', 'pred_sids', 'row_index', 'sid_bucket_item_ids', 'target_candidate_first_source_type', 'target_candidate_source_types', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id', 'target_sid']
cf first-row keys:   ['candidate_details', 'candidate_hit_rank_0_based', 'candidate_item_ids', 'candidate_pool_hit', 'generation_scores', 'history_item_id', 'history_item_sid', 'latency_ms', 'mapped_target_sid', 'pred_sid_valid', 'pred_sids', 'row_index', 'sid_bucket_item_ids', 'target_candidate_first_source_type', 'target_candidate_source_types', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id', 'target_sid']
identity-like common keys: ['history_item_id', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id']
 first-row history_item_id: PASS
 first-row target_in_exact_bucket: PASS
 first-row target_in_prefix: PASS
 first-row target_item_id: PASS

--- mode: p3 ---
text candidates: results/candidates_Industrial_and_Scientific_text_k512_30k_p3_c500_v2/candidates.jsonl exists= True
cf candidates:   results/candidates_Industrial_and_Scientific_cf_k512_30k_p3_c500_v1/candidates.jsonl exists= True
jsonl rows text/cf: 4533 4533 PASS
text first-row keys: ['candidate_details', 'candidate_hit_rank_0_based', 'candidate_item_ids', 'candidate_pool_hit', 'generation_scores', 'history_item_id', 'history_item_sid', 'latency_ms', 'mapped_target_sid', 'pred_sid_valid', 'pred_sids', 'row_index', 'sid_bucket_item_ids', 'target_candidate_first_source_type', 'target_candidate_source_types', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id', 'target_sid']
cf first-row keys:   ['candidate_details', 'candidate_hit_rank_0_based', 'candidate_item_ids', 'candidate_pool_hit', 'generation_scores', 'history_item_id', 'history_item_sid', 'latency_ms', 'mapped_target_sid', 'pred_sid_valid', 'pred_sids', 'row_index', 'sid_bucket_item_ids', 'target_candidate_first_source_type', 'target_candidate_source_types', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id', 'target_sid']
identity-like common keys: ['history_item_id', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id']
 first-row history_item_id: PASS
 first-row target_in_exact_bucket: PASS
 first-row target_in_prefix: PASS
 first-row target_item_id: PASS

====================================================================================================
[DUAL REPORTS] Industrial_and_Scientific
root: results/dual_sid_cf_k512_dedup_Industrial_and_Scientific exists= True

--- mode: exact ---
fusion exists: True
rerank exists: True
schema exists: True
schema top-level keys: ['behavior', 'dedup_by_item', 'sample_alignment', 'text']
schema preview: {"behavior": {"field_mapping": {"candidate_items": "candidate_item_ids", "rank": "list_order_1_based", "sample_id": "row_index", "target_item_id": "target_item_id"}, "mode": "nested_candidate_item_ids", "num_rows": 4533, "path": "results/candidates_Industrial_and_Scientific_cf_k512_30k_exact_c500_v1/candidates.jsonl", "preview": {"num_preview_rows": 3, "path": "results/candidates_Industrial_and_Scientific_cf_k512_30k_exact_c500_v1/candidates.jsonl", "preview": [{"candidate_details_keys": ["all_source_types", "bucket_size", "expansion_level", "item_id", "matched_prefix", "sid_rank_0_based", "source_sid", "source_type"], "candidate_item_ids_len": 20, "keys": ["candidate_details", "candidate_hit_rank_0_based", "candidate_item_ids", "candidate_pool_hit", "generation_scores", "history_item_id", "history_item_sid", "latency_ms", "mapped_target_sid", "pred_sid_valid", "pred_sids", "row_index", "sid_bucket_item_ids", "target_candidate_first_source_type", "target_candidate_source_types", "targe
text_hr20: 0.128612397970439
behavior_hr20: 0.12045003309066843
union_hr20: 0.16346790205162146
behavior_only_hit20: 158
minrank_hr20: 0.13192146481358924
rrf_hr20: 0.1301566291639091
rrf_ndcg20: 0.07596008088193647
rerank top-level keys: ['after_rerank', 'before_rerank', 'delta_after_minus_before', 'inputs', 'leakage_policy', 'outputs', 'topk', 'warnings', 'weights']
rerank preview: {"after_rerank": {"hr@1": 0.03595852636223252, "hr@10": 0.11471431722920802, "hr@20": 0.14515773218619016, "hr@3": 0.07522611956761527, "hr@5": 0.08824178248400617, "hr@50": 0.16346790205162146, "ndcg@1": 0.03595852636223252, "ndcg@10": 0.07278228042800593, "ndcg@20": 0.08044834648858518, "ndcg@3": 0.05908724910261323, "ndcg@5": 0.06437375542263277, "ndcg@50": 0.0842937795970237, "num_samples": 4533}, "before_rerank": {"hr@1": 0.04015001103022281, "hr@10": 0.10522832561217736, "hr@20": 0.1301566291639091, "hr@3": 0.06794617251268475, "hr@5": 0.08030002206044562, "hr@50": 0.16346790205162146, "ndcg@1": 0.04015001103022281, "ndcg@10": 0.06964432952682467, "ndcg@20": 0.07596008088193647, "ndcg@3": 0.05650320517396282, "ndcg@5": 0.061591693422799805, "ndcg@50": 0.08291611993599358, "num_samples": 4533}, "delta_after_minus_before": {"hr@1": -0.00419148466799029, "hr@10": 0.009485991617030662, "hr@20": 0.015001103022281054, "hr@3": 0.007279947054930513, "hr@5": 0.007941760423560554, "hr@50":

--- mode: p3 ---
fusion exists: True
rerank exists: True
schema exists: True
schema top-level keys: ['behavior', 'dedup_by_item', 'sample_alignment', 'text']
schema preview: {"behavior": {"field_mapping": {"candidate_items": "candidate_item_ids", "rank": "list_order_1_based", "sample_id": "row_index", "target_item_id": "target_item_id"}, "mode": "nested_candidate_item_ids", "num_rows": 4533, "path": "results/candidates_Industrial_and_Scientific_cf_k512_30k_p3_c500_v1/candidates.jsonl", "preview": {"num_preview_rows": 3, "path": "results/candidates_Industrial_and_Scientific_cf_k512_30k_p3_c500_v1/candidates.jsonl", "preview": [{"candidate_details_keys": ["all_source_types", "bucket_size", "expansion_level", "item_id", "matched_prefix", "sid_rank_0_based", "source_sid", "source_type"], "candidate_item_ids_len": 181, "keys": ["candidate_details", "candidate_hit_rank_0_based", "candidate_item_ids", "candidate_pool_hit", "generation_scores", "history_item_id", "history_item_sid", "latency_ms", "mapped_target_sid", "pred_sid_valid", "pred_sids", "row_index", "sid_bucket_item_ids", "target_candidate_first_source_type", "target_candidate_source_types", "target_in_
text_hr20: 0.11250827266710788
behavior_hr20: 0.07390249283035517
union_hr20: 0.13876020295609973
behavior_only_hit20: 119
minrank_hr20: 0.10191925876902713
rrf_hr20: 0.11383189940436797
rrf_ndcg20: 0.06337793291546953
rerank top-level keys: ['after_rerank', 'before_rerank', 'delta_after_minus_before', 'inputs', 'leakage_policy', 'outputs', 'topk', 'warnings', 'weights']
rerank preview: {"after_rerank": {"hr@1": 0.03617913081844253, "hr@10": 0.1129494815795279, "hr@20": 0.1414074564306199, "hr@3": 0.0743437017427752, "hr@5": 0.08912420030884624, "hr@50": 0.1972203838517538, "ndcg@1": 0.03617913081844253, "ndcg@10": 0.07248489229224149, "ndcg@20": 0.07978202714897908, "ndcg@3": 0.05866969135703319, "ndcg@5": 0.06477427730855952, "ndcg@50": 0.09067385208697942, "num_samples": 4533}, "before_rerank": {"hr@1": 0.031105228325612178, "hr@10": 0.08890359585263623, "hr@20": 0.11383189940436797, "hr@3": 0.053386278402823735, "hr@5": 0.06794617251268475, "hr@50": 0.15817339510258108, "ndcg@1": 0.031105228325612178, "ndcg@10": 0.057035047634876795, "ndcg@20": 0.06337793291546953, "ndcg@3": 0.044180960399136854, "ndcg@5": 0.05022920804165844, "ndcg@50": 0.07205392053311875, "num_samples": 4533}, "delta_after_minus_before": {"hr@1": 0.005073902492830355, "hr@10": 0.024045885726891675, "hr@20": 0.027575557026251918, "hr@3": 0.020957423339951466, "hr@5": 0.021178027796161486, "hr@50

====================================================================================================
[TEST CSV ALIGNMENT] Office_Products
text_csv: data/Amazon/sid_versions/text_mbk_k512_dedup/Office_Products/test.csv exists= True
cf_csv:   data/Amazon/sid_versions/cf_k512_dedup/Office_Products/test.csv exists= True
len(text), len(cf): 4866 4866
columns(text): ['user_id', 'history_item_title', 'item_title', 'history_item_id', 'item_id', 'history_item_sid', 'item_sid', 'history_item_sid_old', 'item_sid_old']
columns(cf):   ['user_id', 'history_item_title', 'item_title', 'history_item_id', 'item_id', 'history_item_sid', 'item_sid', 'history_item_sid_old', 'item_sid_old']
common identity cols: ['user_id', 'history_item_id', 'item_id']
PASS: row count equal
user_id: PASS equal sequence
history_item_id: PASS equal sequence
item_id: PASS equal sequence

====================================================================================================
[CANDIDATE FILE ALIGNMENT] Office_Products

--- mode: exact ---
text candidates: results/candidates_Office_Products_text_k512_30k_exact_c500_vdual/candidates.jsonl exists= True
cf candidates:   results/candidates_Office_Products_cf_k512_30k_exact_c500_vdual/candidates.jsonl exists= True
jsonl rows text/cf: 4866 4866 PASS
text first-row keys: ['candidate_details', 'candidate_hit_rank_0_based', 'candidate_item_ids', 'candidate_pool_hit', 'generation_scores', 'history_item_id', 'history_item_sid', 'latency_ms', 'mapped_target_sid', 'pred_sid_valid', 'pred_sids', 'row_index', 'sid_bucket_item_ids', 'target_candidate_first_source_type', 'target_candidate_source_types', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id', 'target_sid']
cf first-row keys:   ['candidate_details', 'candidate_hit_rank_0_based', 'candidate_item_ids', 'candidate_pool_hit', 'generation_scores', 'history_item_id', 'history_item_sid', 'latency_ms', 'mapped_target_sid', 'pred_sid_valid', 'pred_sids', 'row_index', 'sid_bucket_item_ids', 'target_candidate_first_source_type', 'target_candidate_source_types', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id', 'target_sid']
identity-like common keys: ['history_item_id', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id']
 first-row history_item_id: PASS
 first-row target_in_exact_bucket: PASS
 first-row target_in_prefix: PASS
 first-row target_item_id: PASS

--- mode: p3 ---
text candidates: results/candidates_Office_Products_text_k512_30k_p3_c500_vdual/candidates.jsonl exists= True
cf candidates:   results/candidates_Office_Products_cf_k512_30k_p3_c500_vdual/candidates.jsonl exists= True
jsonl rows text/cf: 4866 4866 PASS
text first-row keys: ['candidate_details', 'candidate_hit_rank_0_based', 'candidate_item_ids', 'candidate_pool_hit', 'generation_scores', 'history_item_id', 'history_item_sid', 'latency_ms', 'mapped_target_sid', 'pred_sid_valid', 'pred_sids', 'row_index', 'sid_bucket_item_ids', 'target_candidate_first_source_type', 'target_candidate_source_types', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id', 'target_sid']
cf first-row keys:   ['candidate_details', 'candidate_hit_rank_0_based', 'candidate_item_ids', 'candidate_pool_hit', 'generation_scores', 'history_item_id', 'history_item_sid', 'latency_ms', 'mapped_target_sid', 'pred_sid_valid', 'pred_sids', 'row_index', 'sid_bucket_item_ids', 'target_candidate_first_source_type', 'target_candidate_source_types', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id', 'target_sid']
identity-like common keys: ['history_item_id', 'target_in_exact_bucket', 'target_in_prefix', 'target_item_id']
 first-row history_item_id: PASS
 first-row target_in_exact_bucket: PASS
 first-row target_in_prefix: PASS
 first-row target_item_id: PASS

====================================================================================================
[DUAL REPORTS] Office_Products
root: results/dual_sid_cf_k512_dedup_Office_Products exists= True

--- mode: exact ---
fusion exists: True
rerank exists: True
schema exists: True
schema top-level keys: ['behavior', 'dedup_by_item', 'sample_alignment', 'text']
schema preview: {"behavior": {"field_mapping": {"candidate_items": "candidate_item_ids", "rank": "list_order_1_based", "sample_id": "row_index", "target_item_id": "target_item_id"}, "mode": "nested_candidate_item_ids", "num_rows": 4866, "path": "results/candidates_Office_Products_cf_k512_30k_exact_c500_vdual/candidates.jsonl", "preview": {"num_preview_rows": 3, "path": "results/candidates_Office_Products_cf_k512_30k_exact_c500_vdual/candidates.jsonl", "preview": [{"candidate_details_keys": ["all_source_types", "bucket_size", "expansion_level", "item_id", "matched_prefix", "sid_rank_0_based", "source_sid", "source_type"], "candidate_item_ids_len": 20, "keys": ["candidate_details", "candidate_hit_rank_0_based", "candidate_item_ids", "candidate_pool_hit", "generation_scores", "history_item_id", "history_item_sid", "latency_ms", "mapped_target_sid", "pred_sid_valid", "pred_sids", "row_index", "sid_bucket_item_ids", "target_candidate_first_source_type", "target_candidate_source_types", "target_in_exact_buc
text_hr20: 0.14447184545828196
behavior_hr20: 0.1426222770242499
union_hr20: 0.18968351829017674
behavior_only_hit20: 220
minrank_hr20: 0.16152897657213316
rrf_hr20: 0.1621454993834772
rrf_ndcg20: 0.09218864602308983
rerank top-level keys: ['after_rerank', 'before_rerank', 'delta_after_minus_before', 'inputs', 'leakage_policy', 'outputs', 'topk', 'warnings', 'weights']
rerank preview: {"after_rerank": {"hr@1": 0.051787916152897656, "hr@10": 0.1405672009864365, "hr@20": 0.17221537196876285, "hr@3": 0.09391697492807234, "hr@5": 0.11035758323057954, "hr@50": 0.18968351829017674, "ndcg@1": 0.051787916152897656, "ndcg@10": 0.0930878952940814, "ndcg@20": 0.10112432852733567, "ndcg@3": 0.07645799156978543, "ndcg@5": 0.08325038051738694, "ndcg@50": 0.10481748466943923, "num_samples": 4866}, "before_rerank": {"hr@1": 0.0458281956432388, "hr@10": 0.1282367447595561, "hr@20": 0.1621454993834772, "hr@3": 0.08158651870119195, "hr@5": 0.09905466502260583, "hr@50": 0.18968351829017674, "ndcg@1": 0.0458281956432388, "ndcg@10": 0.08355010078873376, "ndcg@20": 0.09218864602308983, "ndcg@3": 0.06688239024279327, "ndcg@5": 0.07409029738559628, "ndcg@50": 0.0979333240324672, "num_samples": 4866}, "delta_after_minus_before": {"hr@1": 0.0059597205096588585, "hr@10": 0.012330456226880393, "hr@20": 0.01006987258528566, "hr@3": 0.012330456226880393, "hr@5": 0.011302918207973703, "hr@50": 0.0

--- mode: p3 ---
fusion exists: True
rerank exists: True
schema exists: True
schema top-level keys: ['behavior', 'dedup_by_item', 'sample_alignment', 'text']
schema preview: {"behavior": {"field_mapping": {"candidate_items": "candidate_item_ids", "rank": "list_order_1_based", "sample_id": "row_index", "target_item_id": "target_item_id"}, "mode": "nested_candidate_item_ids", "num_rows": 4866, "path": "results/candidates_Office_Products_cf_k512_30k_p3_c500_vdual/candidates.jsonl", "preview": {"num_preview_rows": 3, "path": "results/candidates_Office_Products_cf_k512_30k_p3_c500_vdual/candidates.jsonl", "preview": [{"candidate_details_keys": ["all_source_types", "bucket_size", "expansion_level", "item_id", "matched_prefix", "sid_rank_0_based", "source_sid", "source_type"], "candidate_item_ids_len": 171, "keys": ["candidate_details", "candidate_hit_rank_0_based", "candidate_item_ids", "candidate_pool_hit", "generation_scores", "history_item_id", "history_item_sid", "latency_ms", "mapped_target_sid", "pred_sid_valid", "pred_sids", "row_index", "sid_bucket_item_ids", "target_candidate_first_source_type", "target_candidate_source_types", "target_in_exact_bucket",
text_hr20: 0.14180024660912455
behavior_hr20: 0.11344019728729964
union_hr20: 0.177147554459515
behavior_only_hit20: 172
minrank_hr20: 0.1467324290998767
rrf_hr20: 0.1487875051376901
rrf_ndcg20: 0.08677896654572242
rerank top-level keys: ['after_rerank', 'before_rerank', 'delta_after_minus_before', 'inputs', 'leakage_policy', 'outputs', 'topk', 'warnings', 'weights']
rerank preview: {"after_rerank": {"hr@1": 0.051171393341553635, "hr@10": 0.13974517057131114, "hr@20": 0.16872174270448007, "hr@3": 0.09206740649404029, "hr@5": 0.11138512124948623, "hr@50": 0.21249486230990547, "ndcg@1": 0.051171393341553635, "ndcg@10": 0.09210586063603213, "ndcg@20": 0.09945115825890648, "ndcg@3": 0.07498278240781601, "ndcg@5": 0.08288818806344952, "ndcg@50": 0.10809791582780863, "num_samples": 4866}, "before_rerank": {"hr@1": 0.04500616522811344, "hr@10": 0.12145499383477189, "hr@20": 0.1487875051376901, "hr@3": 0.07912042745581586, "hr@5": 0.09432799013563502, "hr@50": 0.19071105630908344, "ndcg@1": 0.04500616522811344, "ndcg@10": 0.07988513596269659, "ndcg@20": 0.08677896654572242, "ndcg@3": 0.06483472351374027, "ndcg@5": 0.07115010530075663, "ndcg@50": 0.09512785292362418, "num_samples": 4866}, "delta_after_minus_before": {"hr@1": 0.0061652281134401965, "hr@10": 0.01829017673653925, "hr@20": 0.01993423756678997, "hr@3": 0.012946979038224421, "hr@5": 0.01705713111385121, "hr@50":
(base) root@autodl-container-a85511aafa-ef91e78f:~/autodl-tmp/projects/MiniOneRec# 









