#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import evaluate_visual_recovery_ppo as base
from v44_multiview_feature_agent import MultiViewFeatureV41Agent


def install_protocol_adapter(accepted_protocol="routed_multiview_feature_adapter_v19", accounting_field="feature_adapter_transitions"):
    original=base.atomic_json; base.VisualAgent=MultiViewFeatureV41Agent
    def annotated(payload,path:Path):
        completion=json.loads((path.parent/"TRAINING_COMPLETE.json").read_text())
        if completion.get("training_protocol") != accepted_protocol: raise ValueError("feature-adapter protocol mismatch")
        local=int(completion[accounting_field])
        if int(payload["checkpoint_global_step"])!=local: raise ValueError("V44 accounting mismatch")
        for key in ("online_ppo_environment_steps","initialization_ppo_environment_steps","ppo_environment_steps","online_protocol_ppo_environment_steps","initialization_protocol_ppo_environment_steps","protocol_ppo_environment_steps","local_bc_dagger_environment_transitions","initialization_bc_dagger_environment_transitions","bc_dagger_environment_transitions"): payload[key]=0
        payload["total_environment_transitions"]=int(completion["total_simulator_transitions"]); payload["protocol_environment_transitions_consumed"]=int(completion["total_simulator_transitions"]); payload["training_protocol"]=completion["training_protocol"]; payload[accounting_field]=local; payload["initialization_simulator_transitions"]=int(completion["initialization_simulator_transitions"]); payload["ppo_accounting_applicable"]=False; payload["domain_label_available_to_actor"]=False; payload["evaluation_source_sha256"]["v44_protocol_adapter"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(); original(payload,path)
    base.atomic_json=annotated


if __name__=="__main__": install_protocol_adapter(); base.main()
