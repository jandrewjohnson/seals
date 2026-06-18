# SEALS

See the [SEALS documentation](https://justinandrewjohnson.com/earth_economy_devstack/seals_overview.html) for full details.

---

## Getting Started

### Environment Setup

See [docs/environment_setup.md](docs/environment_setup.md) for detailed instructions.

Quick start:
```bash
conda create -n hazelbean_env python=3.10
conda activate hazelbean_env
pip install hazelbean seals
```

### Project Setup

1. **Create project folder structure**
   ```bash
   mkdir -p ~/Files/seals/projects/my_project/my_project_dev
   ```
   - `my_project/` - runtime folder (not version controlled)
   - `my_project_dev/` - version controlled code and inputs

2. **Copy and rename the run script**
   ```bash
   cp seals_dev/run_seals_standard.py projects/my_project/my_project_dev/run_my_project.py
   ```

3. **Modify run_my_project.py**
   - Set `project_name = 'my_project'`
   - Set `project_dir = '..'` (parent of _dev folder)
   - Set `input_data_dir = 'input/my_project_input'`
   - Configure scenarios, years, AOI as needed

4. **Create version-controlled input folder**
   ```bash
   mkdir -p my_project_dev/input/my_project_input
   ```

5. **Add your inputs** to `my_project_dev/input/my_project_input/`:
   - [scenarios.csv](docs/scenarios_format.md) - scenario definitions

6. **Run the script**
   ```bash
   conda activate hazelbean_env
   cd my_project_dev
   python run_my_project.py
   ```
   This creates in `my_project/`:
   - `input/` - copied from `my_project_dev/input/my_project_input/`
   - `intermediate/` - SEALS outputs and results
