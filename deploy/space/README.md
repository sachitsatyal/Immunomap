---
title: ImmunoMap
emoji: 🩸
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Immune cell-type annotation for blood single-cell RNA-seq
---

Live demo of [ImmunoMap](https://github.com/sachitsatyal/Immunomap): upload a blood (PBMC) scRNA-seq `.h5ad` file with
raw counts and get lineage / coarse / fine cell types with calibrated confidence and an
Unknown flag. Model: scANVI with reference mapping
([model files](https://huggingface.co/sawcheet/immunomap-scanvi-v1)).
Deployed automatically from GitHub by GitHub Actions after CI passes.
