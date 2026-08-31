#!/usr/bin/env python3
import argparse, hashlib, json, os
from pathlib import Path

def load(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,payload):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_name(f".{path.name}.tmp.{os.getpid()}"); temporary.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(temporary,path)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(); gate=load(args.config); aggregate=load(gate["candidate_aggregate"])
    if aggregate.get("method")!=gate["candidate_method"] or aggregate.get("training_seeds")!=[int(gate["matched_training_seed"])]: raise ValueError("V52 confirmation identity mismatch")
    records=aggregate["records"]; causal=[r for r in records if r["variant"]=="progress_cyclic_shift" and r["condition"]=="intervention"]
    if len(causal)!=1: raise ValueError("V52 confirmation needs one causal record")
    unseen=[r for r in records if r["variant"] not in ("baseline","progress_cyclic_shift")]
    thresholds=gate["thresholds"]; minimum=min(r["variant_safe_success_rate"] for r in unseen); maximum_drop=max(r["baseline_minus_variant_safe_success"] for r in unseen)
    checks={"causal_drop":causal[0]["baseline_minus_variant_safe_success"]>=thresholds["minimum_causal_safe_success_drop"],"causal_lower_bound":causal[0]["paired_cluster_bootstrap_95"][0]>0 if thresholds["require_positive_causal_lower_bound"] else True,"each_unseen_safe_success":minimum>=thresholds["minimum_each_unseen_safe_success"],"each_paired_drop":maximum_drop<=thresholds["maximum_each_paired_safe_success_drop"]}
    payload={"schema_version":1,"gate":gate["name"],"eligible":all(checks.values()),"checks":checks,"observed":{"causal_safe_success_drop":causal[0]["baseline_minus_variant_safe_success"],"causal_paired_cluster_bootstrap_95":causal[0]["paired_cluster_bootstrap_95"],"minimum_unseen_safe_success":minimum,"maximum_paired_safe_success_drop":maximum_drop},"thresholds":thresholds,"source_sha256":{args.config:sha(args.config),gate["candidate_aggregate"]:sha(gate["candidate_aggregate"]),"checker":sha(__file__)},"claim_boundary":gate["claim_boundary"]}; atomic(args.output,payload); print(json.dumps(payload,indent=2,sort_keys=True)); raise SystemExit(0 if payload["eligible"] else 1)

if __name__=="__main__": main()
