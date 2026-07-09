# Analysis-ready input data

These files are the analysis-ready inputs consumed directly by the repository scripts. The repository reproduces the statistical analyses, model validation, tables, and figures from these tables; it does not redownload or preprocess the original remote-sensing and reanalysis products.

| File | Role |
|---|---|
| `Base_Table_2001_2025.csv.bz2` | Canonical grid–year–season modeling table. Bzip2 compression keeps the file below GitHub's 100 MB per-file limit; `pandas.read_csv` detects the compression automatically. |
| `Fire_Event_Table_2001_2025.csv` | Retained MODIS MCD14ML active-fire detections used for temporal and FRP analyses. |
| `Burned_Pixel_Event_Table_2001_2025.csv` | Retained MODIS MCD64A1 burned-pixel records used for temporal and FAI analyses. |
| `ao_monthly_noaa.csv` | Monthly Arctic Oscillation index. |
| `ao_official_fill_2025.csv` | Official values used to replace missing AO entries in late 2025. |
| `pdo_monthly_noaa.csv` | Monthly Pacific Decadal Oscillation index. |
| `nino34_monthly_noaa.csv` | Monthly Niño 3.4 index. |
| `FC_EHSA_all17_country_percentages_wide.csv` | Country-level Fire Count EHSA category summary exported from ArcGIS Pro. |
| `BA_EHSA_all17_country_percentages_wide.csv` | Country-level Burned Area EHSA category summary exported from ArcGIS Pro. |
| `admin1_annual_gistar_class_counts_2001_2025.csv` | Annual administrative-region Gi* class counts exported from the spatial workflow. |

`data_manifest.csv` records the byte size and SHA-256 checksum of every input file. Run `python tools/verify_input_manifest.py` before the analysis.

## Scope of reproducibility

The supplied tables are sufficient to reproduce the analyses contained in this repository. Raw MODIS, ERA5/ERA5-Land, GEFF/CEMS, vegetation, topographic, population, land-use, and OpenStreetMap acquisition and raster preprocessing are described in the manuscript and Supplementary Materials but are not repeated here because those original products are maintained by their respective data providers and may carry separate distribution terms.
