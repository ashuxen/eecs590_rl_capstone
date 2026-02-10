# Cookiecutter Data Science Template Reference

## Template Used

This project was initialized using the **Cookiecutter Data Science** template.

### Template Details

| Property | Value |
|----------|-------|
| **Template Name** | Cookiecutter Data Science |
| **Version** | v2 (2024) |
| **Source** | https://github.com/drivendataorg/cookiecutter-data-science |
| **Documentation** | https://cookiecutter-data-science.drivendata.org/ |
| **License** | MIT |

### Installation Command Used

```bash
# Install cookiecutter via pipx (recommended)
pipx install cookiecutter-data-science

# Create project
ccds
```

### Template Configuration

When running `ccds`, the following options were selected:

| Option | Value |
|--------|-------|
| **project_name** | EECS 590 RL Capstone |
| **repo_name** | eecs590_rl_capstone |
| **module_name** | rl_capstone |
| **author_name** | Ashutosh Kumar |
| **description** | Reinforcement Learning Capstone Project for EECS 590 |
| **python_version_number** | 3.11 |
| **dataset_storage** | none |
| **include_code_scaffold** | Yes |

---

## Project Structure (from template)

The template provides this standardized structure:

```
├── data/
│   ├── external/       <- Data from third party sources
│   ├── interim/        <- Intermediate transformed data
│   ├── processed/      <- Final, canonical data sets
│   └── raw/            <- Original, immutable data
│
├── models/             <- Trained models, predictions, summaries
│
├── notebooks/          <- Jupyter notebooks (naming: number_description.ipynb)
│
├── references/         <- Data dictionaries, manuals, explanatory materials
│
├── reports/
│   └── figures/        <- Generated graphics for reporting
│
├── rl_capstone/        <- Source code (Python module)
│   ├── __init__.py
│   ├── config.py       <- Project configuration
│   ├── dataset.py      <- Dataset utilities
│   ├── features.py     <- Feature engineering
│   ├── modeling/       <- Model training and prediction
│   └── plots.py        <- Visualization code
│
├── Makefile            <- Makefile with convenience commands
├── README.md           <- Project documentation
├── pyproject.toml      <- Project dependencies (modern format)
└── requirements.txt    <- Dependencies (pip format)
```

---

## Customizations Made

The following customizations were made to the template for this RL project:

### Added Directories/Files
- `rl_capstone/environments/` - Environment implementations (WindyChasm)
- `rl_capstone/agents/` - Agent implementations (DPAgent, BaseAgent)
- `rl_capstone/algorithms/` - RL algorithms (Value Iteration, Policy Iteration, etc.)
- `rl_capstone/mdp/` - MDP/MRP classes
- `rl_capstone/visualization/` - Interactive visualizations
- `models/policy_kernel/` - Trained policy storage
- `train_and_save.py` - Training script
- `load_and_use_agent.py` - Agent loading demo

### Modified Files
- `config.py` - Simplified (removed loguru/dotenv dependencies)
- `requirements.txt` - Updated for RL dependencies

---

## Why Cookiecutter Data Science?

1. **Standardized Structure**: Makes project organization consistent and professional
2. **Best Practices**: Separates data, code, and outputs
3. **Reproducibility**: Clear structure for reproducing experiments
4. **Collaboration**: Easy for others to understand and contribute
5. **Industry Standard**: Used by many data science teams

---

## References

- DrivenData. (2024). *Cookiecutter Data Science*. https://cookiecutter-data-science.drivendata.org/
- GitHub Repository: https://github.com/drivendataorg/cookiecutter-data-science

---

*Document created: February 2026*
*Author: Ashutosh Kumar*
