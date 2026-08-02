# Installation

SEALS is installed one way: **hazelbean and the geospatial stack from
conda-forge, then SEALS itself as an editable install from a clone.** Your edits
to the clone take effect immediately, with no reinstall.

SEALS is distributed as **`sealsmodel`** and imported as `seals`. It requires
hazelbean 2.0.0 or later and Python 3.10 or later (hazelbean's floor, set by
NumPy 2).

Only SEALS is cloned. Hazelbean is a normal conda-forge dependency here — if you
also want to *edit* hazelbean, that is the devstack developer install, covered in
the [devstack installation guide](https://justinandrewjohnson.com/earth_economy_devstack/installation.html),
which documents the full set of install paths across the stack.

## 1. Install git and Miniforge/Conda

If not already installed:

- **git** — <https://git-scm.com/downloads>. The install starts from a clone.
- **Miniforge** — <https://github.com/conda-forge/miniforge>. Install for your
  user account only, and accept the option to add it to your `PATH`.

Miniforge ships `mamba` as well as `conda`; the commands below use `mamba` for
installs because conda's solver is very slow on the geospatial stack.

## 2. Install a C/C++ compiler

SEALS compiles a Cython extension (`seals/seals_cython_functions.pyx`) when it is
installed from source, so **a C/C++ compiler is required** — see
[Compiling C/C++ Code](https://justinandrewjohnson.com/earth_economy_devstack/installation.html#compiling-cc-code).

- **macOS:** `xcode-select --install`
- **Windows:** Visual Studio Build Tools, or run `install.bat` from the
  Earth-Economy Devstack repository root
- **Linux:** `gcc` is usually already present

## 3. Create and activate an environment

```bash
conda create -n <your_env> python=3.10
conda activate <your_env>
```

Pick your own environment name — the devstack does not require a particular one.
Always activate it before running SEALS scripts.

## 4. Install hazelbean and the dependency stack from conda-forge

This is the step that makes the editable install below work, so do not skip it —
it is where hazelbean, GDAL, NumPy and the rest of the geospatial stack come
from, as conda builds rather than pip wheels:

```bash
mamba install hazelbean cython libgdal-hdf5
```

This may take 5–10 minutes.

## 5. Clone SEALS into the devstack layout

The repo lives in its own project folder under `~/Files/`
(`C:\Users\<you>\Files\` on Windows). The layout matters: ProjectFlow infers
where to put project directories from it, placing them at
`~/Files/seals/projects/<project_name>/` — outside the repo, so outputs never
land in your working tree.

```bash
mkdir -p ~/Files/seals && cd ~/Files/seals
git clone https://github.com/jandrewjohnson/seals.git
```

## 6. Make the editable install

```bash
cd ~/Files/seals/seals
pip install -e . --no-deps
```

`--no-deps` is deliberate, and required. SEALS declares `hazelbean>=2.0.0` as its
only runtime dependency, and you already have it from step 4. Letting pip resolve
dependencies again tends to replace conda's builds of the geospatial stack (GDAL
and friends) with incompatible wheels.

## 7. Verify

```bash
python -c "import seals; print(seals.__file__)"
```

The path should point inside `~/Files/seals/seals/`, not into `site-packages`. If
it points at `site-packages`, an installed copy is shadowing your clone — remove
it and redo the editable install.

Hazelbean, by contrast, *should* resolve to `site-packages` here: it came from
conda-forge, and that is correct for this install.

## 8. Add other devstack repos (optional)

The same clone-and-editable-install pattern works for the rest of the stack, each
in its own `~/Files/<name>/` project folder:
[global_invest_dev](https://github.com/NatCapTEEMs/global_invest_dev),
[gtap_invest_dev](https://github.com/jandrewjohnson/gtap_invest_dev),
[gtappy_dev](https://github.com/jandrewjohnson/gtappy_dev),
[gep_dev](https://github.com/jandrewjohnson/gep_dev).
All but `global_invest_dev` are private; ask for read access before cloning them.

See the [full installation guide](https://justinandrewjohnson.com/earth_economy_devstack/installation.html)
for the devstack-wide version of these steps.

## Troubleshooting

The devstack guide's
[Common problems](https://justinandrewjohnson.com/earth_economy_devstack/installation.html#common-problems)
section covers the environment-level failures, none of which are SEALS-specific:
you need administrator rights; PowerShell needs `conda init powershell` before
conda works in it; conda may need adding to `PATH` manually; and Windows may show
"Windows protected your PC" on the Miniforge installer (click *More info* → *Run
anyway*).

If `import seals` resolves to `site-packages`, see the verify step in 7 — an
installed copy is shadowing your clone.

## Next

With the environment activated, go to [Quickstart](quickstart.qmd) — it runs the
pared test pipeline and shows where the output lands.

Budget time for the first run rather than the install: SEALS downloads the base
data it needs at runtime from a public bucket, so a correct install is still
followed by a substantial download before you see a result. No credentials are
needed for the default configuration.
