"""Build side-by-side paper/reproduction comparison sheets."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def save_comparison(paper: Path, reproduced: Path, output: Path, figure_name: str, reproduction_label: str | None = None):
    left = mpimg.imread(paper)
    right = mpimg.imread(reproduced)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), constrained_layout=True)
    for ax, image, label in (
        (axes[0], left, f"Original paper {figure_name}"),
        (axes[1], right, reproduction_label or f"Reproduction {figure_name}"),
    ):
        ax.imshow(image)
        ax.axis("off")
        ax.set_title(label, fontsize=13, pad=10)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/paper_reproduction_20261027")
    args = parser.parse_args()
    run = Path(args.run)
    if not run.is_absolute():
        run = ROOT / run
    comparison = run / "comparison"
    crops = comparison / "paper_crops"
    figures = run / "figures"
    comparison.mkdir(exist_ok=True)
    jobs = (
        (crops / "fig1.png", figures / "fig1_pdf_comparison.png", comparison / "fig1_side_by_side", "Fig. 1", None),
        (crops / "fig3.png", figures / "fig3_success_rate.png", comparison / "fig3_side_by_side", "Fig. 3", None),
        (
            crops / "fig4.png",
            figures / "fig4_computational_cost_trial0_paper_scaled.png",
            comparison / "fig4_side_by_side",
            "Fig. 4",
            "Exploratory Fig. 4 (artificial scaling; NOT hardware calibration)",
        ),
        (
            crops / "fig4.png",
            figures / "fig4_computational_cost_trial0.png",
            comparison / "fig4_measured_side_by_side",
            "Fig. 4",
            "Reproduction Fig. 4 (measured time)",
        ),
    )
    for paper, reproduced, output, name, reproduction_label in jobs:
        if not paper.is_file() or not reproduced.is_file():
            raise FileNotFoundError(f"Missing comparison input: {paper} or {reproduced}")
        save_comparison(paper, reproduced, output, name, reproduction_label)
        print(output)
    manifest_path = run / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["outputs"] = {
            str(path.relative_to(run)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(run.rglob("*"))
            if path.is_file() and path.name != "manifest.json" and "team_members_20261028" not in path.parts
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
