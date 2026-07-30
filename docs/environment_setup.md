# SEALS Environment Setup

## 1. Install Miniforge/Conda

If not already installed, download and install [Miniforge](https://github.com/conda-forge/miniforge).

## 2. Create hazelbean environment

```bash
conda create -n hazelbean_env python=3.10
conda activate hazelbean_env
```

## 3. Install dependencies

**From PyPI:**
```bash
pip install hazelbean
pip install seals
```

**Or from source (for development):**
```bash
cd ~/Files/seals/hazelbean_dev
pip install -e .

cd ~/Files/seals/seals_dev
pip install -e .
```

Both methods install all dependencies automatically.

## 4. Activate environment

Always activate before running SEALS scripts:
```bash
conda activate hazelbean_env
```
