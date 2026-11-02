"""ImmunoMap demo: annotate immune cell types in a blood scRNA-seq file.

Run locally:   uv run --group app python app.py      (then open http://127.0.0.1:7860)
Settings (environment variables):
    IMMUNOMAP_BUNDLE     model bundle folder (scripts/export_bundle.py)
    IMMUNOMAP_EXAMPLE    example .h5ad offered by the "Try the example" button
    IMMUNOMAP_MAX_CELLS  larger uploads are randomly subsampled to this many cells
"""

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import anndata as ad
import gradio as gr
import numpy as np
import pandas as pd

from immunomap.inference.annotate import QC_FAILED, Annotator, ensure_raw_counts
from immunomap.models.calibration import UNKNOWN

BUNDLE = os.environ.get("IMMUNOMAP_BUNDLE", "data/models/bundle/immunomap-scanvi-v1")
EXAMPLE = os.environ.get("IMMUNOMAP_EXAMPLE", "data/examples/hao2021_2000cells.h5ad")
MAX_CELLS = int(os.environ.get("IMMUNOMAP_MAX_CELLS", "20000"))

ANNOTATOR = Annotator.from_bundle(BUNDLE)
F1 = ANNOTATOR.manifest.get("macro_f1", {})

INTRO = f"""
# ImmunoMap
Cell-type annotation for **human blood (PBMC) single-cell RNA-seq**. Upload an `.h5ad`
file with raw UMI counts and get, for every cell, a lineage, coarse and fine cell type
(17 coarse / 25 fine), a calibrated confidence, and an **Unknown** flag when the model
is unsure or the cell looks unlike anything in its reference.

Model: scANVI with reference mapping, trained on ~1.2M cells from three studies
(CELLxGENE Census). Macro-F1 on studies it never saw:
Hao 2021 coarse {F1.get("coarse", {}).get("external_ext_hao2021", "n/a")},
Lupus coarse {F1.get("coarse", {}).get("external_ext_lupus", "n/a")}.
Files larger than {MAX_CELLS:,} cells are randomly subsampled. Runs on a free CPU, so
expect about a minute for a few thousand cells.
"""

LIMITS = """
**Limitations.** Human blood only (not tissue, not other species). New cell types close
to a known one (e.g. a rare B-cell state) may be labelled as that type rather than
Unknown. Research use only, not for diagnosis.
"""


def agreement(adata: ad.AnnData, table: pd.DataFrame) -> str:
    """If the file carries author labels (`coarse` column), how often we agree with them."""
    if "coarse" not in adata.obs:
        return ""
    truth = adata.obs["coarse"].astype(object).to_numpy()
    called = np.array([isinstance(t, str) for t in truth]) & ~np.isin(
        table["coarse"].to_numpy(), [UNKNOWN, QC_FAILED]
    )
    if not called.any():
        return ""
    agree = (truth[called] == table["coarse"].to_numpy()[called]).mean()
    return f"\n\nAgreement with the file's own `coarse` labels (called cells): **{agree:.1%}**"


def run(path: str | None) -> tuple[str, pd.DataFrame, pd.DataFrame, str]:
    if not path:
        raise gr.Error("Upload an .h5ad file or click 'Try the example'.")
    try:
        adata = ensure_raw_counts(ad.read_h5ad(path))
    except ValueError as error:
        raise gr.Error(str(error)) from error
    except Exception as error:
        raise gr.Error(f"Could not read this file as .h5ad: {error}") from error
    note = ""
    if adata.n_obs > MAX_CELLS:
        rng = np.random.default_rng(0)
        adata = adata[np.sort(rng.choice(adata.n_obs, MAX_CELLS, replace=False))].copy()
        note = f" (randomly subsampled to {MAX_CELLS:,})"

    start = time.time()
    table, summary = ANNOTATOR.annotate(adata)
    seconds = time.time() - start

    counts = table["coarse"].value_counts().rename_axis("cell type").reset_index(name="cells")
    unknown = (table["coarse"] == UNKNOWN).mean()
    missing = summary["model_genes_missing"]
    text = (
        f"**{summary['cells']:,} cells**{note} annotated in {seconds:.0f} s. "
        f"{summary['passed_qc']:,} passed QC; {unknown:.1%} Unknown at the coarse level."
        + (
            f" Note: {missing:,} of {summary['model_genes']:,} model genes were missing."
            if missing
            else ""
        )
        + (
            f"\n\n**Warning:** this file has only {adata.n_vars:,} genes. QC expects the full "
            "transcriptome (all detected genes), so many cells may fail QC."
            if adata.n_vars < 10_000
            else ""
        )
        + agreement(adata, table)
    )
    out = Path(tempfile.mkdtemp()) / "immunomap_annotations.csv"
    table.rename_axis("cell_id").to_csv(out, lineterminator="\n")
    preview = table.rename_axis("cell_id").reset_index().head(500).round(3)
    return text, counts, preview, str(out)


def build() -> Any:
    with gr.Blocks(title="ImmunoMap") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            upload = gr.File(label="Your .h5ad file (raw counts)", file_types=[".h5ad"])
            with gr.Column():
                annotate = gr.Button("Annotate", variant="primary")
                example = gr.Button("Try the example (2,000 cells, Hao 2021)")
        summary = gr.Markdown()
        chart = gr.BarPlot(x="cell type", y="cells", title="Cells per coarse type", height=360)
        table = gr.Dataframe(label="First 500 cells (download the CSV for all)", wrap=True)
        download = gr.File(label="All annotations (CSV)")
        gr.Markdown(LIMITS)
        outputs = [summary, chart, table, download]
        annotate.click(run, inputs=upload, outputs=outputs)
        example.click(lambda: run(EXAMPLE), inputs=None, outputs=outputs)
    return demo


if __name__ == "__main__":
    build().queue(default_concurrency_limit=1).launch()
