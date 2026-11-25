# DS203 Nation Clustering

Final report and sequential Python code for clustering countries/economies using eleven World Bank indicators for 2022.

Atharva Airen (25B1207), Aaryan Bhattad (25B1231), Gandhar Vartak (25B1238).

## Run

Open a terminal in the folder containing the scripts:

```bash
python -m pip install -r requirements.txt
python ds203_nation_clustering.py
```

Tested with Python 3.13.5. The bundled `data/wdi_raw_2022.csv` lets the analysis run offline. The script uses relative paths, prints statistical summaries and country-count tables, writes processed data to `data/`, and saves eleven figures to `figures/`.

## Method

The analysis follows the teammate's original preprocessing: exclude economies missing three or more indicators, then fill remaining gaps with regional indicator medians among retained economies, using a global median only as a fallback. In this snapshot, all 82 filled cells use regional medians. GDP per capita is log10 transformed, and all eleven features are standardized.

PCA describes explained variance and component coefficients. K-Means and Ward clustering use all eleven standardized numerical features. IncomeGroup and LendingType are excluded from preprocessing and fitting. Region is used for imputation and later interpretation; it is not included as an additional model feature, so geographical comparisons are partly influenced by preprocessing.

K-Means searches k=2 through 8 using within-cluster sum of squares, seed 42 and 20 restarts. The final k=4 model uses 30 restarts. Four groups provide descriptive resolution rather than a uniquely optimal partition. Ward uses the same features and four groups, with a cut derived from its merge distances. Country-count contingency tables compare methods and income groups. Euclidean distance to the assigned K-Means centroid identifies unusual economies.

## Results

| Quantity | Result |
| --- | --- |
| Raw / retained / excluded economies | 217 / 174 / 43 |
| K-Means sizes, groups 0 / 1 / 2 / 3 | 59 / 33 / 3 / 79 |
| Groups | Advanced / Lower development / High inflation / Intermediate |
| PC1 / PC2 / PC3 variance | 43.15% / 13.30% / 9.60% |
| First three / four PCs cumulative variance | 66.05% / 74.78% |
| Components exceeding 80% cumulative variance | 5, explaining 81.84% |
| Ward group sizes | 16 / 89 / 45 / 24 |
| Ward four-group cut | 16.7308 |

K-Means isolates Lebanon, Sudan and Zimbabwe as a high-inflation group. Ward places them with thirteen other economies. The comparison uses membership counts and numerical profiles.

The source reports Central African Republic life expectancy as 18.818 years. This unusually low value is retained and flagged as a source-quality concern. Missingness exclusions, regional imputation and the snapshot's contemporary income categories limit interpretation. Imputed values should not be described as observed measurements.

## Files

- `report_final.pdf`: final four-page report, in the GitHub repository.
- `ds203_nation_clustering.py`: sequential analysis with compact formatting and no code comments.
- `fetch_data.py`: original-style optional World Bank downloader. Running it from this folder overwrites `data/wdi_raw_2022.csv`; fetching is unnecessary for the bundled analysis.
- `requirements.txt`: Python dependencies.
- `data/wdi_raw_2022.csv`: frozen input snapshot.
- `data/data_dictionary.csv` and `data/data_source.json`: variable definitions and provenance.

PCA and the clustering comparison are requested in the project brief; the supplied instructor clarification permits hierarchical clustering libraries.
