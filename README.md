# eBay Auction Price Prediction

This repository contains the selected eBay/Online Auctions research workstream. It evaluates whether competition, time dynamics and bidder feedback add predictive value beyond observed prices. It excludes unrelated streetwear/StockX work, environments, duplicate preliminary auction experiments, confidential project references and personal course submissions.

## Dataset

The input is `auction.csv` from [Online Auctions Dataset](https://www.kaggle.com/datasets/onlineauctions/online-auctions-dataset). The provider credits the companion site for *Modeling Online Auctions*, Wolfgang Jank and Galit Shmueli (2010).

- 10,681 bid rows and 628 auctions; Xbox 149, PDA 343, watches 136.
- Exclude one PDA auction with inconsistent opening prices: 627 auctions, 10,653 bid rows.
- The standalone Xbox dataset duplicates the Xbox subset and is not independent validation.
- `bid` is described by the provider as a **proxy bid**; it should not be equated automatically with a publicly displayed standing price.
- Raw CSV files contain bidder handles, including email-like values. They remain local and are not committed.
- The provider lists the license as **Unknown** (checked 21 October 2026). This repository supplies source metadata and a local downloader rather than republishing that file.

See [data/README.md](data/README.md) for the source fingerprint and fields. No license for the original dataset is granted by this repository.

## Environment and reproduction

Windows PowerShell, Python 3.12:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-auction.txt
.\.venv\Scripts\python.exe scripts/download_online_auctions.py
.\.venv\Scripts\python.exe scripts/run_auction_feature_experiments.py prepare
.\.venv\Scripts\python.exe scripts/run_auction_feature_experiments.py xbox
.\.venv\Scripts\python.exe scripts/run_auction_feature_experiments.py pda
.\.venv\Scripts\python.exe scripts/run_auction_feature_experiments.py watch
.\.venv\Scripts\python.exe scripts/run_auction_feature_experiments.py finish
.\.venv\Scripts\python.exe scripts/summarize_auction_feature_experiments.py
```

`requirements-auction.txt` records direct package versions used locally; it is not a full transitive dependency lock.

Optionally run `build_auction_meeting_figures.py` followed by `restructure_auction_meeting.py` to regenerate the presentation figures and speaking notes. Word notes are generated locally; no PDF is required. Figure 06 is supplied as an anonymised presentation preview.

## Experimental design

- One auction-level snapshot at 25%, 50% and 75% elapsed duration; 50% is primary.
- Only bids at or before the cutoff enter features; auctions with no bids are retained.
- Products modeled separately with identical duration-stratified 5-fold splits repeated 3 times.
- Fixed Ridge alpha=10; random forest 100 trees, depth 5, minimum leaf 5.
- B0 simple benchmarks; B1 opening conditions; B2 observed prices.
- B3/B4/B5 add competition, dynamics or feedback separately to B2; B6 combines them.
- Group-removal ablations and limited exploratory B7 interactions.
- Genuine new bidders are distinguished from all recent active bidders.
- MAE primary, RMSE/R-squared supplementary; exploratory paired uncertainty checks.

No calendar start timestamps are available: these are auction-level cross-validation results, not future-calendar tests. Recorded feedback timing and real-time visibility of proxy bids remain limitations.

## Main findings

Observed prices provide the main gains. Extra behavioural signals do not show consistent positive incremental value across products, models and cutoffs. Time dynamics shows small candidate gains at the midpoint, with positive-gain uncertainty intervals crossing zero. This is an exploratory predictive study, not causal identification or fraud detection.

- [Presentation figures](docs/meeting_20261021/)
- [Aggregate metrics and diagnostics](outputs/auction_feature_experiments/)
- `results_summary.xlsx` includes aggregate results only. Per-auction features/predictions and the original larger workbook are generated locally but excluded from Git.

## Git scope

`.gitignore` is an explicit allowlist. New files are ignored unless deliberately added to the allowed scope. This prevents `git add .` from including unrelated work or private submission files.

No files were deleted to prepare this repository. Original PDFs, literature review, personal progress forms and the sponsor-context midterm Word report remain local under `docs/planning_submission/` or `docs/`. GitHub repository: https://github.com/crescentIsland/ftec5910 .

Only the six selected presentation images under docs/ are tracked. Reports, notes, Word/PDF files and the local selection guide remain excluded. The analysis scripts regenerate local reports when needed.
