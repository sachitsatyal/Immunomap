# ImmunoMap

Immune cell-type annotation for blood scRNA-seq, delivered as a website and REST API.
Upload a file, get cell-type labels with calibrated confidence, an "Unknown" flag
for unfamiliar cells, and an interactive UMAP.

> Research tool, not a diagnostic device. Work in progress.

## Quick start (development)
    uv sync
    uv run pytest

See [PRODUCT_SPEC.md](PRODUCT_SPEC.md) for what the product does.