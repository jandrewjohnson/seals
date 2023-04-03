# SEALS

SEALS is under active development and will change much as it moves from a personal research library to supported model. This repository is the public, open-source version of SEALS and contains releases that correspond to different published manuscripts. However, because SEALS is progressing quickly, we have created a new repository for the development version of SEALS. If you would like to be involved in the development of SEALS, please reach out to jajohns@umn.edu. The public version is several releases behind the dev version (at least until we can fund real software developers!).

To run a minimal version of the model:
> python run_test_seals.py

In order for the above line to work, you will need to set the project directory and data directory lines in run_test_seals.py. For seals to work, you will have to pro vide the necessary input data. However, if you want to replicate the model on the global data, feel free to reach out to us to coordinate transfer of the data (it's pretty big at 300gb). 

To run a full version of the model, copy run_test_seals.py to a new file (i.e., run_seals.py) and set p.test_mode = False. You may also want to specify a new project directory to keep different runs separate.

## Installation

The following installation steps are very bloated and are basically a "kitchen-sink" approach to making sure every possibly-relevant library is intalled. Use at your own discretion.

-   Install Mambaforge from https://github.com/conda-forge/miniforge#mambaforge
-   For convenience, during installation, I select yes for "Add Mambaforge to my PATH environment Variable"
-   (PC) Open the Miniforge Prompt (search for it in the start menu) or (Mac) just type "mamba init"
-   Create a new mamba environment with the following commands (here it is named 8222env1):

`mamba create -n gtap_invest_env -c conda-forge`

-   Activate the environment

`mamba activate gtap_invest_env`

-   Install libraries using conda command:

`mamba install -c conda-forge natcap.invest geopandas rasterstats netCDF4 cartopy xlrd markdown qtpy qtawesome plotly descartes pygeoprocessing taskgraph cython rioxarray dask google-cloud-datastore google-cloud-storage aenum anytree statsmodels openpyxl seaborn twine pyqt ipykernel imageio pandoc`

-   And then finally, install non-conda distributions via pip:

`pip install mglearn pandoc datascience hazelbean`

After your python environment is setup, use Git to clone the gtap_invest repository. Open run_test_gtap_pnas.py to replicate the model run used to generate results from the current submission. See the section "code structure" for more details on how to proceed further.

## Numpy errors

If numpy throws "wrong size or changes size binary": upgrade numpy at the end of the installation process. See for details: https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp


## Hazelbean and Project Flow

SEALS relies on the Hazelbean project, available via PIP. Hazelbean is a collection of geospatial processing tools based on gdal, numpy, scipy, cython, pygeoprocessing, taskgraph, natcap.invest, geopandas and many others to assist in common spatial analysis tasks in sustainability science, ecosystem service assessment, global integrated modelling assessment,  natural capital accounting, and/or calculable general equilibrium modelling.

Note that for hazelbean to work, your computer will need to be configured to compile Cython files to C code. This workflow is tested in a Python  3.10, 64 bit Windows environment. It should work on other system environments, but this is not yet tested. 

One key component of Hazelbean is ProjectFlow, which manages directories, base_data, parallel computation and other details. ProjectFlow defines a tree of tasks that can easily be run in parallel where needed and keeping track of task-dependencies. ProjectFlow borrows heavily in concept (though not in code) from the taskgraph Python library but adds a predefined file structure suited to research and exploration tasks.

### Running a ProjectFlow model

Models will define a run file, written in Python, e.g., run.py, which will initialize the project flow object. This is the only place where user supplied (possibly absolute but can be relative) path is stated. The p ProjectFlow object is the one global variable used throughout all parts of hazelbean.

``` python
import hazelbean as hb

if __name__ == '__main__':
    p = hb.ProjectFlow(r'C:/Users/jajohns/Files/Research/cge/gtap_invest/projects/pnas_manuscript')
```

In a multi-file setup, in the run.py you will need to import different scripts, such as main.py i.e.:

``` python
import visualizations.main
```

The script file main.py can have whatever code, but in particular can include "task" functions. A task function, shown below, takes only p as an agrument and returns p (potentially modified). It also must have a conditional (if p.run_this:) to specify what always runs (and is assumed to run trivially fast, i.e., to specify file paths) just by nature of having it in the task tree and what is run only conditionally (based on the task.run attribute, or optionally based on satisfying a completed function.)

``` python
def example_task_function(p):
    """Fast function that creates several tiny geotiffs of gaussian-like kernels for later use in ffn_convolve."""

    if p.run_this:
        for i in computationally_intensive_loop:
            print(i)
```

**Important Non-Obvious Note**

Importing the script will define function(s) to add "tasks", which take the ProjectFlow object as an argument and returns it after potential modification.

``` python
def add_all_tasks_to_task_tree(p):
    p.generated_kernels_task = p.add_task(example_task_function)
```



## More information
See the author's personal webpage, https://justinandrewjohnson.com/ for more details about the underlying research.
