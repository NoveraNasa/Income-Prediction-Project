# update_readme.py
from __future__ import annotations
import json
from pathlib import Path
from textwrap import dedent

RESULTS_START = "<!-- RESULTS:START -->"
RESULTS_END   = "<!-- RESULTS:END -->"

def _fmt_float(x, nd=3):
    return "—" if x is None or (isinstance(x, float) and (x != x)) else f"{x:.{nd}f}"

def build_results_markdown(metrics_json_path: Path, outputs_dir: Path) -> str:
    """
    Build a markdown block summarizing metrics and linking confusion/ROC plots
    that were saved during training (one per model).
    """
    data = json.loads(metrics_json_path.read_text(encoding="utf-8"))
    results = data.get("results", {})
    best = data.get("best", {})
    best_name = best.get("best_model_name", "—")
    best_f1   = best.get("best_f1", None)

    # Table header
    md = []
    md.append("### 📊 Results (auto-generated)")
    md.append("")
    md.append(f"**Best model by F1:** `{best_name}` (F1 = {_fmt_float(best_f1)})")
    md.append("")
    md.append("| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Best Params | CV F1 |")
    md.append("|---|---:|---:|---:|---:|---:|---|---:|")

    # Rows
    for name, r in results.items():
        acc = _fmt_float(r.get("accuracy"))
        pre = _fmt_float(r.get("precision"))
        rec = _fmt_float(r.get("recall"))
        f1  = _fmt_float(r.get("f1"))
        auc = _fmt_float(r.get("roc_auc"))
        bp  = r.get("best_params")
        bp_str = ", ".join(f"`{k.replace('model__','')}`={v}" for k, v in (bp or {}).items()) or "—"
        cvf1 = _fmt_float(r.get("cv_f1"))
        md.append(f"| {name} | {acc} | {pre} | {rec} | {f1} | {auc} | {bp_str} | {cvf1} |")

    # Plots
    md.append("")
    md.append("#### 🖼 Diagnostic Plots")
    for name in results.keys():
        conf = outputs_dir / f"confusion_{name}.png"
        roc  = outputs_dir / f"roc_{name}.png"
        conf_md = f"![Confusion {name}]({conf.as_posix()})" if conf.exists() else f"_No confusion matrix for {name}_"
        roc_md  = f"![ROC {name}]({roc.as_posix()})"         if roc.exists()  else f"_No ROC for {name}_"
        md.append(f"- **{name}**  \n  {conf_md}  \n  {roc_md}")
    md.append("")
    return "\n".join(md)

def update_readme_with_results(readme_path: Path, metrics_json_path: Path, outputs_dir: Path):
    """
    Replaces content between RESULTS markers in README with fresh results markdown.
    If markers don't exist, appends a new Results section at the end.
    """
    new_block = build_results_markdown(metrics_json_path, outputs_dir)
    new_section = "\n".join([RESULTS_START, "", new_block, "", RESULTS_END])

    if readme_path.exists():
        text = readme_path.read_text(encoding="utf-8")
    else:
        text = "# Income Prediction Project\n\n"

    if (RESULTS_START in text) and (RESULTS_END in text):
        prefix = text.split(RESULTS_START)[0]
        suffix = text.split(RESULTS_END)[-1]
        updated = prefix + new_section + suffix
    else:
        # append at end
        updated = text.rstrip() + "\n\n" + new_section + "\n"

    readme_path.write_text(updated, encoding="utf-8")
    print(f"README updated with results between markers: {readme_path}")
