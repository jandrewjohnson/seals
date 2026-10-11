# SEALS

SEALS (Spatial Economic Allocation Landscape Simulator) downscales coarse or
regional land-use change projections to fine-resolution (300 meter) LULC maps
globally in less than 30 minutes on a laptop.

All documentation are generated from QMDs in  `docs/` and are hosted at the [Seals Homepage](https://justinandrewjohnson.com/seals/), and the [User Guide](https://justinandrewjohnson.com/seals/user_guide/index.html), which includes pages for installation , quickstart, a guided first run, the scenarios CSV reference, and several 
model linkage exercises. 


## Install Summary

See [Installation](https://justinandrewjohnson.com/seals/user_guide/installation.html) for detailed instructions.

SEALS is on conda-forge as `sealsmodel` (the name `seals` was taken; you still
`import seals`). The package is prebuilt for Windows, macOS and Linux and brings
hazelbean and the geospatial stack with it, so no compiler or clone is needed:

```bash
conda create -n <your_env>
conda activate <your_env>
mamba install sealsmodel
```

### Developer install (editable, from a clone)

Only needed if you intend to change SEALS itself. It requires a C/C++ compiler
(see the installation page) and the dependency stack from conda-forge:

1. Get the dependencies, without a packaged SEALS that would shadow your clone:

```bash
conda create -n <your_env>
conda activate <your_env>
mamba install hazelbean cython libgdal-hdf5
```

2. Clone this repository:

```bash
git clone https://github.com/jandrewjohnson/seals
```

3. In the root of the clone, with `<your_env>` activated, make the editable install:

```bash
pip install -e . --no-deps
```

`--no-deps` is required: the dependencies are already in the environment from
conda, and letting pip resolve them again replaces conda's GDAL stack with
incompatible wheels. If `sealsmodel` is already installed in the environment,
remove it first with `conda remove sealsmodel --force` (keeps the dependencies).

## Run it

A run file is launched from a project folder, not from inside the package. With
the conda-forge install, make one and copy in a run file and scenarios CSV as
shown in the [Quickstart](https://justinandrewjohnson.com/seals/user_guide/quickstart.html);
with a clone, the shipped run file works in place:

```bash
conda activate <your_env>
cd seals/seals
python run_seals.py
```

Outputs land in a Project file just outside of the repo, e.g. at 
`~/Files/seals/projects/seals/`, containing `input/`, `intermediate/`,
`output/`, and one folder per task. Missing base data is downloaded on demand.

Run it a second time and it finishes almost immediately — every task is behind an
existence check, so completed work is skipped.

For a faster first run, `run_seals_test.py` uses the same task tree with
a pared scenarios CSV (baseline plus one BAU, one projection year, Rwanda).

## Repository layout

| path | what |
|------|------|
| `seals/run_seals.py` | the reference run; copy this to start a project |
| `seals/run_seals_test.py` | its pared variant |
| `seals/input_template/` | tracked definition CSVs, read in place (a project's `input/` holds only overrides) |
| `seals/seals_initialize_project.py` | task-tree builders and CSV hydration |
| `seals/seals_main.py`, `seals_tasks.py`, `seals_generate_base_data.py`, … | task functions |
| `seals/seals_utils.py` | helpers |
| `seals/old_run_files/` | unsupported historical run files, kept for reference only |
| `seals_tests/` | tests |

# Hazelbean, ProjectFlow and the Earth-Economy Devstack

SEALS relies on Hazelbean 2.1.0 or later, which is a Python package that provides high-performance
geospatial functions and ProjectFlow, which enables parallel computation of a task tree. Hazelbean,
and the rest of the Earth-Economy Devstack is documented in [the devstack docs](https://justinandrewjohnson.com/earth_economy_devstack).
