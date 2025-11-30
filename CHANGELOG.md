# Changelog

## [2025-11-30] - Project Restructuring

### Added
- Standard project structure following ML/NLP best practices
- `requirements.txt` with all project dependencies
- `.gitignore` for version control
- `PROJECT_STRUCTURE.md` documentation
- `CHANGELOG.md` for tracking changes

### Changed
- Reorganized data files:
  - Raw data moved to `data/raw/` (separated by dataset)
  - Processed data moved to `data/processed/`
- Moved scripts to `scripts/` directory
- Moved reports to `reports/` directory
- Created `src/` package structure with submodules:
  - `src/data/` for data processing
  - `src/models/` for model definitions
  - `src/utils/` for utilities
- Updated script paths to reflect new structure
- Updated README.md with new project structure

### Removed
- Duplicate files from root directory
- Empty directories after reorganization

### Structure
```
.
├── data/
│   ├── raw/          # Original datasets
│   └── processed/    # Cleaned data
├── src/              # Source code package
├── scripts/          # Standalone scripts
├── notebooks/        # Jupyter notebooks
├── models/           # Saved models
└── reports/         # Documentation
```

