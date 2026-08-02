# SEALS

SEALS (Spatial Economic Allocation Landscape Simulator) downscales coarse or
regional land-use change projections to fine-resolution (300 meter) LULC maps
globally in less than 30 minutes on a laptop.

All documentation are generated from QMDs in  `docs/` and are hosted at the [Seals Homepage](https://justinandrewjohnson.com/seals/), and the [User Guide](https://justinandrewjohnson.com/seals/user_guide/index.html), which includes pages for installation , quickstart, a guided first run, the scenarios CSV reference, and several 
model linkage exercises. 


## Install Summary

See [Installation](https://justinandrewjohnson.com/seals/user_guide/installation.html) for detailed instructions.

1. First install Hazelbean via condaforge

```bash
conda create -n <your_env>
conda activate <your_env>
conda install hazelbean
```

2. Clone this repository:

```bash
git clone https://github.com/jandrewjohnson/seals
```

3. Install cloned SEALS repository. In the root of the repository, with <your_env> activated, run:

```bash
pip install -e . --no-deps
```

`--no-deps` assumes the dependencies are already in the environment from conda. 
Note that running SEALS on a Windows PC requires having a C/C++ compiler installed, which is described in more details on the installation page.

## Run it

```bash
conda activate <your_env>
cd seals/seals
python run_seals_standard.py
```

Outputs land in a Project file just outside of the repo, e.g. at 
`~/Files/seals/projects/seals_standard/`, containing `input/`, `intermediate/`,
`output/`, and one folder per task. Missing base data is downloaded on demand.

Run it a second time and it finishes almost immediately — every task is behind an
existence check, so completed work is skipped.

For a faster first run, `run_seals_standard_test.py` uses the same task tree with
a pared scenarios CSV (baseline plus one BAU, one projection year, Rwanda).

## Repository layout

| path | what |
|------|------|
| `seals/run_seals_standard.py` | the reference run; copy this to start a project |
| `seals/run_seals_standard_test.py` | its pared variant |
| `seals/input_template/` | tracked definition CSVs, seeded into each project's `input/` |
| `seals/seals_initialize_project.py` | task-tree builders and CSV hydration |
| `seals/seals_main.py`, `seals_tasks.py`, `seals_generate_base_data.py`, … | task functions |
| `seals/seals_utils.py` | helpers |
| `seals/old_run_files/` | unsupported historical run files, kept for reference only |
| `seals_tests/` | tests |

# Hazelbean, ProjectFlow and the Earth-Economy Devstack

SEALS relies on Hazelbean 2.0.0 or later, which is a Python package that provides high-performance
geospatial functions and ProjectFlow, which enables parallel computation of a task tree. Hazelbean,
and the rest of the Earth-Economy Devstack is documented in [the devstack docs](https://justinandrewjohnson.com/earth_economy_devstack).
