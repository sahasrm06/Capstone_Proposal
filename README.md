# Reducing false alarm review in PCB manufacturing

Week 1 proposal foundation for QM640 Data Analytics Capstone. Student: Murali Sahasranaman.

Business decision: identify low-risk groups within the AOI alarm queue and evaluate potential manual review reduction while measuring detection of confirmed defects. Timestamp groups are board proxies, not verified board IDs.

## Four research questions

1. Does group defect prevalence differ across dominant inspection types?
2. Does a frozen retained-review rule capture more than 90% of confirmed defect groups?
3. Does the rule identify more than 10% of alarm groups as candidates to avoid review, conditional on RQ2?
4. Does group defect prevalence differ between early and late production periods?

The 90% detection threshold is an academic screening criterion, not an approved production release standard. Models and final evaluation will be implemented in later project weeks. No model outcomes are claimed here.

## Install and reproduce the Week 1 checks

Use Python 3.11 or newer in an isolated environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/audit_data.py
python scripts/sample_sizes.py
```

On Windows, activate `.venv\Scripts\activate` instead of the `source` command. Download requires about 334 MB; allow roughly 2 GB available RAM for the audit.

The manifest fixes the source version and SHA256 hashes. The download script refuses to overwrite a conflicting local file. The audit checks all source hashes, 440,274 rows, 78 columns including the exported index, and 39,742 timestamp groups. It writes results/audit.json and results/dominant_type_counts.csv. The sample script reproduces N1=1,611; N2=10,220 group equivalents (225 positive evaluation groups); N3=3,900; N4=10,870. Nminimum=max(...)=10,870, with actual use of all 39,742 groups and separate evaluation-event constraints.

## Organization

- data/SOURCES.md and source_manifest.json: original public access and provenance.
- scripts/: executed audit and calculation code.
- docs/data_dictionary.csv: all raw fields and derived analysis definitions.
- results/: Week 1 real-data audit and generated calculations.

## Data restrictions and submission status

Used the original Mendeley source, not Kaggle. No synthetic training records, surveys, or interviews are used. See data/SOURCES.md for the provider notice. Raw data are intentionally not uploaded. The course template requests complete data and code on GitHub; 

