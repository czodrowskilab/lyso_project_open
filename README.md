# Explainable Machine Learning and Lysosomotropism

This repository contains the code used to perform the modelling and analysis for the paper:

**Identification of Lysosomotropism using Explainable Machine Learning and Morphological Profiling Cell Painting Data**  
Aishvarya Tandon, Anna Santura, Dr. Axel Pahl, Prof. Dr. Herbert Waldmann, Prof. Dr. Paul Czodrowski

Example Jupyter Notebooks are provided to demonstrate the use of the code.

## Installation and Usage

This code was developed and tested on Linux (Ubuntu 22.04.3) and macOS (Big Sur 11.7.10). 

The repository can be cloned using the following command:

```bash
git clone https://github.com/czodrowskilab/lyso_project_open
cd lyso_project_open
```

Then, create a new mamba or conda environment using the provided environment file `environment_xml.yml` (for XML part):

```bash
mamba env create -f environment_xml.yml
mamba activate lyso_xml-env
``` 

Replace `mamba` with `conda` if you don't have mamba installed.

To use `lyso_project` as a package in a Jupyter notebook or a Python file, you need to append the path of the `lyso_project` directory to your system path. Here's how:

```python
import sys

# Prevent Python from writing .pyc files to the disk
sys.dont_write_bytecode = True

# Append the path of the lyso_project directory to your system path
sys.path.append("/path/to/lyso_project")
```

Replace `/path/to/lyso_project` with the actual path of the `lyso_project` directory on your system.

In addition to the packages listed in `environment_xml.yml`, [X-FP](https://github.com/czodrowskilab/x-fp) is also used in this project. Please follow the instructions on the X-FP repository for installation and usage.

## lyso_project

### preprocess_cp_dataset.py
Contains functions to preprocess the internal CP dataset. 

### descriptor_maker.py
Contains various useful functions to generate various molecular fingerprints, and calculate RDKit descriptors. Also contains minor utility functions.

### utils.py
Module to create model objects, perform cross-validation, and calculate metrics.

Key components include:

- **ModelRunner class**: This class is used to create model objects for each input type. It has methods for setting the model's input data, which can include training data, training labels, test data, test labels. Additionally, it has methods for saving models, and writing logs. 

### toml_log_reader_with_plotter.py
Contains functions to read the log files generated by the `ModelRunner` class, and plot the results.


Please refer to the inline comments in the code for more detailed information about the functionality of each method.
