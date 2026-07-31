# SEALS

SEALS (Spatial Economic Allocation Landscape Simulator) downscales coarse or
regional land-use change projections to fine-resolution LULC maps — 300 m
globally in about an hour on a laptop.

The **[SEALS User Guide](docs/index.qmd)** in `docs/` is the full documentation —
install, quickstart, a guided first run, the scenarios CSV reference, and MAgPIE
coupling. It renders as a website and as a single PDF. For how run files are
structured across the Earth-Economy Devstack, see the
[run-file conventions](https://justinandrewjohnson.com/earth_economy_devstack/conventions.html).

---

## Install

SEALS is distributed as **`sealsmodel`** and imported as `seals`. It requires
[hazelbean](https://github.com/jandrewjohnson/hazelbean_dev) 1.9.0 or later,
which supplies `ProjectFlow`.

```bash
conda create -n <your_env> python=3.10
conda activate <your_env>
pip install hazelbean sealsmodel
```

Pick your own environment name — the devstack does not require a particular one.

### From source (development)

```bash
cd ~/Files/hazelbean/hazelbean_dev
pip install -e .

cd ~/Files/seals/seals_dev
pip install -e .
```

---

## Run it

```bash
conda activate <your_env>
cd seals_dev/seals
python run_seals_standard.py
```

Outputs do **not** land in the repo. Because the run file lives inside a cloned
repo, ProjectFlow places the project directory just outside it, at
`~/Files/seals/projects/seals_standard/`, containing `input/`, `intermediate/`,
`output/`, and one folder per task. Missing base data is downloaded on demand.

Run it a second time and it finishes almost immediately — every task is behind an
existence check, so completed work is skipped.

For a faster first run, `run_seals_standard_test.py` uses the same task tree with
a pared scenarios CSV (baseline plus one BAU, one projection year, Rwanda).

---

## Start your own project

1. **Copy the run file** into your project repo as `run_<project>.py`:

   ```bash
   cp seals_dev/seals/run_seals_standard.py my_project_dev/run_my_project.py
   ```

2. **Edit the two lines in `__main__`** — that is the whole of the configuration:

   ```python
   p = hb.ProjectFlow(project_name='my_project', run_mode='check')
   p.scenario_definitions_filename = 'my_project_scenarios.csv'
   ```

   `run_mode` selects how much prior work is reused: `'check'` (default) resumes
   in the stable project dir; `'fresh_intermediate'` rebuilds all computation but
   keeps `input/` (test projects only); `'full'` timestamps a fresh dir.

3. **Put your definition CSVs in `input_template/`** next to the run file. That
   directory is tracked in git; ProjectFlow copies anything missing into the
   project's untracked `input/` on first run and never overwrites your working
   copy — so machine-specific values you fill in survive re-runs. See
   [docs/scenarios_format.md](docs/scenarios_format.md) for the columns.

4. **Run it** from your project repo:

   ```bash
   python run_my_project.py
   ```

### Variants are their own file, never a fork

To run the same pipeline differently — a smoke test, another AOI — write a second
run file that *imports* the pipeline instead of copying it.
`run_seals_standard_test.py` is the worked example:

```python
import hazelbean as hb
from run_my_project import run_project

if __name__ == '__main__':
    p = hb.ProjectFlow(project_name='my_project_test', run_mode='check')
    p.scenario_definitions_filename = 'my_project_scenarios_test.csv'
    # p.tasks_to_skip = ['stitched_lulc_simplified_scenarios']
    run_project(p)
```

What differs is data and placement — never code. Copying the run file and editing
a line instead means the two silently diverge the first time you change an input.

---

## Anatomy of a run file

Three parts, in every run file in the devstack:

- `build_task_tree(p)` — what the pipeline is. Nothing but `add_task` calls; for
  stock SEALS it delegates to `seals_initialize_project.build_standard_task_tree(p)`.
- `run_project(p)` — how it is executed. It sets what no variant ever changes
  (base data location, processing resolution); the caller sets what a variant
  might (project name, run mode, scenarios CSV).
- an `if __name__ == '__main__':` guard that builds and configures the
  ProjectFlow, then calls `run_project(p)`. Importing a run file must never start
  a run — that is what makes the variant pattern above possible.

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

## Environment

See [docs/environment_setup.md](docs/environment_setup.md) for detailed setup,
including the C/C++ compiler needed to build the Cython extensions.
