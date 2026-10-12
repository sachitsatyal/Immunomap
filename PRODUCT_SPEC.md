# ImmunoMap — Product Spec

## User
A scientist or bioinformatician at a biotech/pharma team who has blood (PBMC)
scRNA-seq data and needs fast, consistent, trustworthy immune cell-type labels.

## What the product is
A deployed website (and REST API behind it): upload a file, get annotated cells back.

## Inputs
- `.h5ad` (AnnData) or 10x output (`.h5`), human PBMC/blood, raw counts
- Max file size: 500 MB (v1)

## Outputs
- QC summary: low-quality cells and doublets flagged
- Cell-type label per cell (coarse + fine, Cell Ontology IDs)
- Calibrated confidence per cell, and an "Unknown" flag
- Interactive UMAP of the user's cells on the reference atlas
- Downloads: annotated `.h5ad`, CSV of labels, PDF report
- Every result records model, reference and pipeline versions

## Website pages
- Home: what it does and a demo dataset
- Upload: drag-and-drop, validation messages, job progress
- Results: UMAP, label table, confidence histogram, QC summary, downloads
- API docs

## API (v1)
- `POST /jobs`: upload a file, returns a job ID
- `GET /jobs/{id}`: job status
- `GET /jobs/{id}/results`: labels, confidence, download links
- `GET /models`: available model versions
- `GET /health`, `GET /ready`, `GET /metrics`

## Scope
- v1: runs locally with Docker Compose; CI on every pull request
- v2: deployed on AWS (EKS) with Kubernetes, monitoring, drift checks, rollback

## Out of scope
Diagnosis or clinical use; non-blood tissues; non-human data.
