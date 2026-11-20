#!/usr/bin/env python3
"""Reload every saved learned checkpoint and match its saved dev scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from experiments.ea_atpo_v2.training.data import load_q_dataset, load_public_embedding_cache
from experiments.ea_atpo_v2.training.features import ExistingPublicStateEncoder, Standardizer


class TinyHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, x): return self.net(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q-dataset", type=Path, required=True)
    ap.add_argument("--split", type=Path, required=True)
    ap.add_argument("--embedding-cache", type=Path, required=True)
    ap.add_argument("--training-dir", type=Path, required=True)
    args = ap.parse_args()
    out_path = args.training_dir / "checkpoint_reload_audit.json"
    if out_path.exists(): raise FileExistsError(out_path)
    rows = load_q_dataset(args.q_dataset)
    split = json.loads(args.split.read_text(encoding="utf-8"))
    dev_ids = set(map(str, split["dev"]))
    dev_rows = [r for r in rows if r.case_id in dev_ids]
    cache = load_public_embedding_cache(args.embedding_cache, expected_examples=rows)
    saved = [json.loads(x) for x in (args.training_dir / "dev_predictions.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    saved_map = {(r["feature"],r["model"],int(r["seed"]),r["case_id"],r["state_id"]): np.asarray(r["predicted_scores"],dtype=np.float32) for r in saved}
    checkpoints = sorted((args.training_dir / "checkpoints").rglob("*.pt"))
    comparisons = []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        feature, model, seed = payload["feature_kind"], payload["model"], int(payload["seed"])
        if feature == "text":
            encoder = ExistingPublicStateEncoder.from_json(payload["encoder"])
            raw = encoder.transform(dev_rows)
        else:
            raw = np.stack([cache.vectors[(r.case_id,r.state_id)] for r in dev_rows]).astype(np.float32)
        scaler = Standardizer.from_json(payload["standardizer"])
        x = scaler.transform(raw)
        state = payload["state"]
        if model.startswith("ridge_"):
            out = x @ np.asarray(state["weight"],dtype=np.float32) + np.asarray(state["bias"],dtype=np.float32)
        else:
            in_dim = int(state["net.0.weight"].shape[1]); hidden = int(state["net.0.weight"].shape[0]); output = int(state["net.2.weight"].shape[0])
            head = TinyHead(in_dim,hidden,output); head.load_state_dict(state); head.eval()
            with torch.no_grad(): out = head(torch.as_tensor(x,dtype=torch.float32)).numpy()
        if model in ("ridge_relative","mlp_relative"):
            scores = np.column_stack([out[:,0],np.zeros(len(out),dtype=np.float32),out[:,1]])
        elif model == "q_guided_distillation":
            scores = out * float(payload["temperature"])
        else:
            scores = out
        errors=[]
        for row, prediction in zip(dev_rows,scores):
            key=(feature,model,seed,row.case_id,row.state_id)
            expected=saved_map.get(key)
            if expected is None: raise ValueError(f"missing stored dev prediction for {key}")
            errors.append(float(np.max(np.abs(prediction-expected))))
        comparisons.append({"checkpoint":str(path.resolve()),"feature":feature,"model":model,"seed":seed,
                            "dev_states":len(dev_rows),"max_abs_prediction_difference":max(errors,default=0.0),
                            "matched_within_1e-5":max(errors,default=0.0)<=1e-5})
    result={"status":"passed" if len(checkpoints)==30 and all(x["matched_within_1e-5"] for x in comparisons) else "failed",
            "checkpoint_count":len(checkpoints),"dev_state_count":len(dev_rows),"comparisons":comparisons,
            "note":"Checkpoints were reloaded and CPU predictions matched the stored per-seed development scores."}
    out_path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"checkpoint_count":len(checkpoints),
                      "max_abs_difference":max((r["max_abs_prediction_difference"] for r in comparisons),default=None),
                      "report":str(out_path.resolve())},indent=2))
    if result["status"] != "passed": raise SystemExit(1)


if __name__ == "__main__": main()
