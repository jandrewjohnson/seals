# SEALS

SEALS is under active development and will change much as it moves from a personal research library to supported model. We have submitted this code for publication but are waiting on reviews. 

To run a minimal version of the model:
> python run_test_seals.py

In order for the above line to work, you will need to set the project directory and data directory lines in run_test_seals.py. To obtain the base_data necessary, see the SEALS manuscript for the download link. 

To run a full version of the model, copy run_test_seals.py to a new file (i.e., run_seals.py) and set p.test_mode = False. You may also want to specify a new project directory to keep different runs separate.

## Hazelbean
SEALS relies on the Hazelbean project, available via PIP. Hazelbean is a collection of geospatial processing tools based on gdal, numpy, scipy, cython, pygeoprocessing, taskgraph, natcap.invest, geopandas and many others to assist in common spatial analysis tasks in sustainability science, ecosystem service assessment, global integrated modelling assessment,  natural capital accounting, and/or calculable general equilibrium modelling.

 Note that for hazelbean to work, your computer will need to be configured to compile Cython files to C code. This workflow is tested in a Python  3.10, 64 bit Windows environment. It should work on other system environments, but this is not yet tested. 



## Installation

Pip installing Hazelbean will attempt to install required libraries, but many of these must be compiled for your computer. You can solve each one manually for your chosen opperating system, or you can use these Anaconda-based steps here:

- Install Anaconda3 with the newest python version (tested at python 3.6.3)
- Install libraries using conda command: "conda install -c conda-forge geopandas"
- Install libraries using conda command: "conda install -c conda-forge rasterstats"
- Install libraries using conda command: "conda install -c conda-forge netCDF4"
- Install libraries using conda command: "conda install -c conda-forge cartopy"
- Install libraries using conda command: "conda install -c conda-forge xlrd markdown"
- Install libraries using conda command: "conda install -c conda-forge qtpy qtawesome"
- Install libraries using conda command: "conda install -c conda-forge plotly descartes"
- Install libraries using conda command: "conda install -c conda-forge pygeoprocessing"
- Install libraries using conda command: "conda install -c conda-forge taskgraph"
- Install libraries using conda command: "conda install -c conda-forge cython"
- Install libraries using conda command: "conda install -c conda-forge rioxarray"
- Pip install anytree

And then finally,
- Install hazelbean with "pip install hazelbean"

## Numpy errors

If numpy throws "wrong size or changes size binary": upgrade numpy at the end of the installation process. See for details: https://stackoverflow.com/questions/66060487/valueerror-numpy-ndarray-size-changed-may-indicate-binary-incompatibility-exp


## More information
See the author's personal webpage, https://justinandrewjohnson.com/ for more details about the underlying research.

## Project Flow

One key component of Hazelbean is that it manages directories, base_data, etc. using a concept called ProjectFlow. ProjectFlow defines a tree of tasks that can easily be run in parallel where needed and keeping track of task-dependencies. ProjectFlow borrows heavily in concept (though not in code) from the task_graph library produced by Rich Sharp but adds a predefined file structure suited to research and exploration tasks. 
