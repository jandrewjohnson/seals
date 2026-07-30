# SEALS Environment Setup

Two paths: a **user install** if you just want to run SEALS, or a **developer
install** from a GitHub clone if you want to edit SEALS (or hazelbean) and have
your edits take effect immediately.

SEALS is distributed as **`sealsmodel`** and imported as `seals`. It requires
hazelbean 1.9.0 or later.

## 1. Install Miniforge/Conda

If not already installed, download and install
[Miniforge](https://github.com/conda-forge/miniforge).

## 2. Create and activate an environment

```bash
conda create -n <your_env> python=3.10
conda activate <your_env>
```

Pick your own environment name — the devstack does not require a particular one.
Always activate it before running SEALS scripts.

## 3a. User install

```bash
pip install hazelbean sealsmodel
```

Dependencies are installed automatically. This is enough to run
`run_seals_standard.py` and to start your own project from it.

## 3b. Developer install from a GitHub clone

Use this if you cloned the repos and want your edits to take effect without
reinstalling. SEALS compiles a Cython extension
(`seals/seals_cython_functions.pyx`), so **a C/C++ compiler is required** — see
[Compiling C/C++ Code](https://justinandrewjohnson.com/earth_economy_devstack/installation.html#compiling-cc-code)
(`xcode-select --install` on macOS; Visual Studio Build Tools on Windows).

**Clone into the devstack layout.** Each repo lives in its own project folder
under `~/Files/` (`C:\Users\<you>\Files\` on Windows). The layout matters:
ProjectFlow infers where to put project directories from it, placing them at
`~/Files/seals/projects/<project_name>/` — outside the repo, so outputs never
land in your working tree.

```bash
mkdir -p ~/Files/hazelbean && cd ~/Files/hazelbean
git clone https://github.com/jandrewjohnson/hazelbean_dev.git

mkdir -p ~/Files/seals && cd ~/Files/seals
git clone https://github.com/jandrewjohnson/seals_dev.git
```

**If you already installed hazelbean from conda-forge**, remove it first so the
editable install is the one that gets imported. Use `conda remove`, not
`pip uninstall` — pip leaves behind directories that still shadow the source:

```bash
conda remove hazelbean --force
```

This drops the conda-forge build but keeps its dependencies.

**Make the editable installs**, hazelbean first — SEALS imports it:

```bash
cd ~/Files/hazelbean/hazelbean_dev
pip install -e . --no-deps

cd ~/Files/seals/seals_dev
pip install -e . --no-deps
```

`--no-deps` is deliberate: the dependencies came from conda in step 2, and
letting pip resolve them again tends to replace conda's builds of the geospatial
stack (GDAL and friends) with incompatible wheels.

**Verify:**

```bash
python -c "import hazelbean, seals; print(hazelbean.__file__); print(seals.__file__)"
```

Both paths should point inside `~/Files/...`, not into `site-packages`. If either
points at `site-packages`, an installed copy is shadowing your clone — remove it
and redo the editable install.

## 4. Add other devstack repos (optional)

The same clone-and-editable-install pattern works for the rest of the stack, each
in its own `~/Files/<name>/` project folder:
[gtap_invest_dev](https://github.com/jandrewjohnson/gtap_invest_dev),
[global_invest_dev](https://github.com/jandrewjohnson/global_invest_dev),
[gtappy_dev](https://github.com/jandrewjohnson/gtappy_dev),
[gep_dev](https://github.com/jandrewjohnson/gep_dev).

See the [full installation guide](https://justinandrewjohnson.com/earth_economy_devstack/installation.html)
for the devstack-wide version of these steps.
