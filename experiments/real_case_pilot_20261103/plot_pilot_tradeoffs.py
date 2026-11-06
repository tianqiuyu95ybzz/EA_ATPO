#!/usr/bin/env python3
"""Plot the measured cost-sensitivity frontier from summary.csv."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GROUP_RE = re.compile(r"lambdaE=([0-9.eE+-]+);lambdaP=([0-9.eE+-]+)")
METRICS = (
    "selected_accuracy",
    "selected_defer_rate",
    "selected_expert_tokens",
    "selected_patient_turns",
    "selected_total_tokens",
)
ACTION_NAMES = ("ANSWER", "ASK", "CONSULT", "DEFER")


def load_points(summary_path: Path) -> list[dict[str, float]]:
    grouped: dict[str, dict[str, str]] = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("section") == "policy_tradeoff" and row.get("metric") in METRICS:
                grouped.setdefault(row["group"], {})[row["metric"]] = row["value"]
                grouped[row["group"]][f"{row['metric']}_ci95_low"] = row.get("ci95_low", "")
                grouped[row["group"]][f"{row['metric']}_ci95_high"] = row.get("ci95_high", "")

    points = []
    for group, metrics in grouped.items():
        match = GROUP_RE.fullmatch(group)
        if not match or any(metric not in metrics for metric in METRICS):
            continue
        point = {
            "lambda_expert_per_1k_tokens": float(match.group(1)),
            "lambda_patient_per_turn": float(match.group(2)),
            **{metric: float(metrics[metric]) for metric in METRICS},
        }
        for metric in METRICS:
            for suffix in ("ci95_low", "ci95_high"):
                raw = metrics.get(f"{metric}_{suffix}", "")
                point[f"{metric}_{suffix}"] = float(raw) if raw else point[metric]
        points.append(point)
    if not points:
        raise ValueError(f"no policy tradeoff points found in {summary_path}")
    return sorted(points, key=lambda p: (p["lambda_expert_per_1k_tokens"], p["lambda_patient_per_turn"]))


def save_points(points: list[dict[str, float]], output_path: Path) -> None:
    fields = [
        "lambda_expert_per_1k_tokens", "lambda_patient_per_turn",
        *METRICS,
        *(f"{metric}_{suffix}" for metric in METRICS for suffix in ("ci95_low", "ci95_high")),
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)


def load_action_points(summary_path: Path) -> list[dict[str, float | str]]:
    table: dict[tuple[str, str], dict[str, str]] = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("section") in ("action", "cost"):
                table[(row["group"], row["metric"])] = row
    actions = []
    for action in ACTION_NAMES:
        accuracy = table[(action, "accuracy")]
        tokens = table[(action, "total_tokens_per_rollout")]
        actions.append({
            "action": action,
            "accuracy": float(accuracy["value"]),
            "accuracy_ci95_low": float(accuracy["ci95_low"]),
            "accuracy_ci95_high": float(accuracy["ci95_high"]),
            "total_tokens_per_rollout": float(tokens["value"]),
            "total_tokens_ci95_low": float(tokens["ci95_low"]),
            "total_tokens_ci95_high": float(tokens["ci95_high"]),
        })
    return actions


def save_action_points(points: list[dict[str, float | str]], output_path: Path) -> None:
    fields = (
        "action", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high",
        "total_tokens_per_rollout", "total_tokens_ci95_low", "total_tokens_ci95_high",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(points)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plot_action_cost_curve(
    points: list[dict[str, float | str]], output_dir: Path
) -> None:
    colors = {"ANSWER": "#4c78a8", "ASK": "#f58518", "CONSULT": "#54a24b", "DEFER": "#e45756"}
    ordered = sorted(points, key=lambda point: float(point["total_tokens_per_rollout"]))
    xs = [float(point["total_tokens_per_rollout"]) for point in ordered]
    ys = [float(point["accuracy"]) for point in ordered]
    fig, ax = plt.subplots(figsize=(6.7, 4.8), layout="constrained")
    ax.plot(xs, ys, color="#65758b", linewidth=1.2, alpha=0.75, zorder=1)
    offsets = {"ANSWER": (9, 8), "ASK": (8, 8), "CONSULT": (8, -26), "DEFER": (8, 8)}
    for point in ordered:
        action = str(point["action"])
        x, y = float(point["total_tokens_per_rollout"]), float(point["accuracy"])
        xerr = [[max(0.0, x - float(point["total_tokens_ci95_low"]))],
                [max(0.0, float(point["total_tokens_ci95_high"]) - x)]]
        yerr = [[max(0.0, y - float(point["accuracy_ci95_low"]))],
                [max(0.0, float(point["accuracy_ci95_high"]) - y)]]
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr, fmt="o", markersize=8,
            markeredgecolor="black", markeredgewidth=0.6,
            color=colors[action], ecolor=colors[action], capsize=3,
            elinewidth=1.1, label=action, zorder=2,
        )
        ax.annotate(action, (x, y), xytext=offsets[action], textcoords="offset points", fontsize=9)
    ax.set_xlabel("Mean total inference tokens per rollout")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        "Accuracy–cost by fixed action\n"
        "Case-clustered 95% bootstrap intervals; failed-call costs may be undercounted",
        fontsize=12,
    )
    y_low = min(float(point["accuracy_ci95_low"]) for point in points)
    y_high = max(float(point["accuracy_ci95_high"]) for point in points)
    y_pad = max(0.025, (y_high - y_low) * 0.12)
    ax.set_ylim(max(0.0, y_low - y_pad), min(1.0, y_high + y_pad))
    x_low = min(float(point["total_tokens_ci95_low"]) for point in points)
    x_high = max(float(point["total_tokens_ci95_high"]) for point in points)
    x_pad = max(20.0, (x_high - x_low) * 0.06)
    ax.set_xlim(max(0.0, x_low - x_pad), x_high + x_pad)
    ax.grid(True, color="#d9dfe7", linewidth=0.7, alpha=0.8)
    fig.savefig(output_dir / "accuracy_cost_curve.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / "accuracy_cost_curve.svg", bbox_inches="tight")
    plt.close(fig)


def plot_curve(
    points: list[dict[str, float]],
    *,
    x_metric: str,
    x_label: str,
    title: str,
    png_path: Path,
    svg_path: Path,
) -> None:
    grouped: dict[tuple[float, float], list[dict[str, float]]] = {}
    for point in points:
        key = (round(point[x_metric], 9), round(point["selected_accuracy"], 9))
        grouped.setdefault(key, []).append(point)
    unique = [members[0] for _, members in sorted(grouped.items())]

    fig, ax = plt.subplots(figsize=(6.7, 4.8), layout="constrained")
    xs = [point[x_metric] for point in unique]
    ys = [point["selected_accuracy"] for point in unique]
    ax.plot(xs, ys, color="#4c78a8", linewidth=1.5, alpha=0.85, zorder=1)
    for index, point in enumerate(unique):
        x = point[x_metric]
        y = point["selected_accuracy"]
        xerr = [[max(0.0, x - point[f"{x_metric}_ci95_low"])],
                [max(0.0, point[f"{x_metric}_ci95_high"] - x)]]
        yerr = [[max(0.0, y - point["selected_accuracy_ci95_low"])],
                [max(0.0, point["selected_accuracy_ci95_high"] - y)]]
        ax.errorbar(
            x, y, xerr=xerr, yerr=yerr,
            fmt="o", markersize=8 + min(4, len(grouped[(round(x, 9), round(y, 9))])),
            markeredgecolor="black", markeredgewidth=0.65,
            color="#4c78a8", ecolor="#4c78a8", elinewidth=1.1,
            capsize=3, zorder=2,
        )
        members = grouped[(round(x, 9), round(y, 9))]
        patient_values = sorted({member["lambda_patient_per_turn"] for member in members})
        patient_label = ", ".join(f"{value:g}" for value in patient_values)
        ax.annotate(
            f"λP={patient_label}\n{len(members)} settings",
            (x, y), xytext=(0, 14 if index % 2 == 0 else -32),
            textcoords="offset points", ha="center", fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Selected policy accuracy")
    ax.set_title(title)
    y_low = min(point["selected_accuracy_ci95_low"] for point in unique)
    y_high = max(point["selected_accuracy_ci95_high"] for point in unique)
    y_pad = max(0.02, (y_high - y_low) * 0.12)
    ax.set_ylim(max(0.0, y_low - y_pad), min(1.0, y_high + y_pad))
    x_low = min(point[f"{x_metric}_ci95_low"] for point in unique)
    x_high = max(point[f"{x_metric}_ci95_high"] for point in unique)
    x_pad = max(0.01, (x_high - x_low) * 0.06)
    ax.set_xlim(x_low - x_pad, x_high + x_pad)
    ax.grid(True, color="#d9dfe7", linewidth=0.7, alpha=0.8)
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    points = load_points(args.summary)
    action_points = load_action_points(args.summary)
    save_points(points, args.output_dir / "tradeoff_curve_points.csv")
    save_action_points(action_points, args.output_dir / "action_cost_accuracy_points.csv")
    plot_action_cost_curve(action_points, args.output_dir)
    plot_curve(
        points,
        x_metric="selected_defer_rate",
        x_label="Selected DEFER rate",
        title="Accuracy–deferral sensitivity\nPost-hoc oracle over observed outcomes; not deployable",
        png_path=args.output_dir / "accuracy_deferral_curve.png",
        svg_path=args.output_dir / "accuracy_deferral_curve.svg",
    )
    pilot_dir = Path(__file__).resolve().parent
    analysis_manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "Post-run plotting/report formatting only; no model inference or outcome recomputation.",
        "analysis_script": str(pilot_dir / "plot_pilot_tradeoffs.py"),
        "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
        "report_script": str(pilot_dir / "make_pilot_report.py"),
        "report_script_sha256": file_sha256(pilot_dir / "make_pilot_report.py"),
        "requirements_input": str(pilot_dir / "PILOT_ANALYSIS_REQUIREMENTS.in"),
        "requirements_lock": str(pilot_dir / "PILOT_ANALYSIS_REQUIREMENTS.lock.txt"),
        "requirements_lock_sha256": file_sha256(pilot_dir / "PILOT_ANALYSIS_REQUIREMENTS.lock.txt"),
        "matplotlib_version": importlib.metadata.version("matplotlib"),
        "analysis_packages_added_after_inference": {
            name: importlib.metadata.version(name)
            for name in ("matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver", "pyparsing")
        },
        "input_sha256": {
            args.summary.name: file_sha256(args.summary),
            "action_value_analysis.csv": file_sha256(args.output_dir / "action_value_analysis.csv"),
        },
        "output_sha256": {
            name: file_sha256(args.output_dir / name)
            for name in (
                "tradeoff_curve_points.csv", "action_cost_accuracy_points.csv",
                "accuracy_cost_curve.png", "accuracy_cost_curve.svg",
                "accuracy_deferral_curve.png", "accuracy_deferral_curve.svg",
            )
        },
    }
    (args.output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(action_points)} fixed-action points, {len(points)} policy-sensitivity points, and four curve files to {args.output_dir}")


if __name__ == "__main__":
    main()
