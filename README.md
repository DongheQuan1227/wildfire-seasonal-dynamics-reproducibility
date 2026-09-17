# Reproducibility package: seasonal forest-fire dynamics in the China–North Korea–Russia transboundary region

This repository contains the analysis-ready input tables and source code associated with the published article **“How Do Multi-Factor Interactions Shape the Seasonal Dynamics of Forest Fires across the China–North Korea–Russia Transboundary Region?”** The workflow covers 2001–2025 and reproduces the temporal analyses, FAI-based fire-season delineation, atmospheric-circulation analysis, hotspot summaries, count-regression models, Hurdle Random Forest models, temporal and spatial out-of-fold validation, OOF SHAP interpretation, unified model comparison, and seasonal SEM analyses.

## Reproducibility scope

The repository starts from the analysis-ready tables in `00_Input_data/`. It reproduces the statistical analyses, model outputs, tables, and figures generated from those tables. Acquisition and raster preprocessing of the original MODIS, ERA5/ERA5-Land, GEFF/CEMS, vegetation, terrain, population, land-use, and OpenStreetMap products are documented in the manuscript and Supplementary Materials but are not redownloaded by this repository.

## Quick start on Windows

1. Install **Python 3.11** and **R 4.1.3 or a compatible later release**.
2. Open a terminal in the repository folder.
3. Install Python packages:

```bat
python -m pip install -r requirements.txt
```

4. Install R packages:

```bat
Rscript tools\install_r_packages.R
```

5. Double-click **`run_all.bat`**, or run:

```bat
python run_all.py --profile full --resume --verify
```

The runner executes every script in the required order, stores console logs under `run_logs/`, and validates key outputs against the manuscript values at the end.

## Conda alternative

```bash
conda env create -f environment.yml
conda activate wildfire-reproducibility
Rscript tools/install_r_packages.R
python run_all.py --profile full --resume --verify
```

## Runner options

```bash
python run_all.py --profile full --list
python run_all.py --preflight-only
python run_all.py --profile descriptive --verify
python run_all.py --profile full --from-step 08.07 --resume --verify
python run_all.py --profile full --to-step 10.08
```

Profiles:

- `descriptive`: temporal, FAI, FRP, country, circulation, EHSA, and administrative-region analyses.
- `modeling`: regression, Random Forest, unified model comparison, and SEM.
- `full`: all analyses.

The complete workflow is computationally intensive. Model selection, spatial validation, bootstrap resampling, SHAP calculation, and SEM uncertainty propagation can require many hours. Reserve substantial free disk space for generated checkpoints, predictions, and figures.

## Repository structure

```text
00_Input_data/                  Analysis-ready input tables and checksums
01_Temporal_patterns/           Annual, monthly, and seasonal temporal analysis
02_FAI_season_delineation/      Data-driven fire-season delineation
03_FRP_trends/                  Fire Radiative Power trend analysis
04_Country_season_trends/       Country- and season-specific totals and trends
05_Atmospheric_circulation/     AO, PDO, and Niño 3.4 analysis
06_EHSA_country_table/          Country-level EHSA summary table
07_Admin1_hotspot_recurrence/   Administrative-region hotspot recurrence
08_Regression_modeling/         NB1/ZINB1 model development and validation
09_RF_modeling/                 Hurdle Random Forest, validation, and OOF SHAP
10_Model_comparison/            Unified OOF comparison and figures
11_SEM_mechanism_analysis/      Seasonal SEM preparation, fitting, and effects
tools/                          Integrity, package, and result-validation utilities
validation/                     Expected manuscript values used by the validator
```

## Input-data integrity

The large canonical base table is stored as `Base_Table_2001_2025.csv.bz2` to remain below GitHub’s 100 MB single-file limit. Pandas reads the compressed table directly; no manual extraction is required.

Verify all supplied inputs before running:

```bash
python tools/verify_input_manifest.py
```

## Output validation

After the workflow completes:

```bash
python tools/validate_reproducibility.py
```

The validator checks input hashes and selected manuscript values, including FAI peaks, FRP trends, country shares, Niño 3.4 relationships, administrative hotspot recurrence, model-comparison improvements, and SEM total effects. During an incomplete run, use:

```bash
python tools/validate_reproducibility.py --allow-partial
```

## Software notes

The manuscript’s archived Python environment used Python 3.11 with NumPy, pandas, SciPy, statsmodels, scikit-learn, Matplotlib, and SHAP. The SEM workflow used R 4.1.3 with `glmmTMB` and supporting packages. Exact rendering can vary slightly across operating systems if Times New Roman is unavailable, although numerical results should remain unchanged.

## Public-release notes

- No API keys, passwords, personal user directories, or machine-specific absolute paths are included.
- Script and output-directory names were standardized for public release without changing analytical calculations.
- Machine-specific defaults were replaced by repository-relative path detection.
- The analytical logic and model settings were not intentionally changed. See `CHANGELOG.md`.

## Citation

This repository provides the reproducibility package associated with the following published article:

**How Do Multi-Factor Interactions Shape the Seasonal Dynamics of Forest Fires across the China–North Korea–Russia Transboundary Region?**  
*Ecological Informatics* (2026).  
https://doi.org/10.1016/j.ecoinf.2026.104049

If you use the code, analysis-ready data, or workflow provided in this repository, please cite the associated article above.

Citation metadata are also provided in `CITATION.cff`.

## License

The software code in this repository is released under the MIT License.
The included analysis-ready data and derived data products remain subject
to the terms and conditions of their original data providers.

