# MaLTOSe - Machine Learning Tailoring Organic Semiconductors
-- Machine-learning code for predicting molecular properties --

## Installation from source

### Option 1: Create a python virtual environment with venv and pip

This will provide the dependencies for the maltose code. Choose the path for the virtual environment according to your preferences:

```
VENVPATH=$HOME/venvs/maltose-pip
```

Set up a new python environment:

```
python3 -m venv $VENVPATH
```

Activate the new environment:

```
source $VENVPATH/bin/activate
```

Upgrade the pip package manager:

```
python3 -m pip install --upgrade pip
```

To install the requirements, execute the commands described below in the folder where also this readme file is located.
The requirements.txt contains at least one package (torch-scatter) depends on torch already being installed. Therefore, if you do not have torch installed yet, install the basic requirement (including torch, but not torch-scatter) first:

```
python3 -m pip install -r requirements_0.txt
```

Then, make sure all the other requirements are installed, too:

```
python3 -m pip install -r requirements.txt
```

Build the maltose package:

```
python3 -m build
```

Install the maltose package:

```
pip install dist/example_package_maltose-0.0.1-py3-none-any.whl
```

You're ready to go!

### Option 2: Create a conda virtual environment

Some users prefer conda over pip because it does not only install python packages, but also allows installing a different python version. Somtimes conda is better at resolving dependencies than pip.

Create and activate a new conda environment:

```
VENVNAME="maltose-conda"
conda create -n "${VENVNAME}" -c conda-forge python=3.9.7
conda activate "${VENVNAME}"
```

Install dependencies:

```
conda install -c conda-forge --file requirements_conda.txt
```

Build the maltose wheel:

```
python3 -m build
```

Install the maltose wheel via pip:

```
pip install --no-dependencies dist/maltose-0.0.1-py3-none-any.whl
```

You're ready to go!

## Running experiments

Remark on file locations: As long as no proper handling of paths is implemented, it is recommend to execute all scripts from the same parent directory of the scripts folder.

To prepare the training data, execute:

```
python3 scripts/primary_data/prepare_qm9.py
python3 scripts/primary_data/prepare_alchemy.py
python3 scripts/primary_data/prepare_oe62.py
python3 scripts/primary_data/prepare_hopv.py
```

Next, prepare the unified train/validation/test split by running:

```
python3 scripts/primary_data/unified_split.py
```

To train a maltose model, run:

```
python3 scripts/multitask_model_v01.py
```

Note that this may take some days or even weeks, depending on your hardware. If available, run the training on a GPU:

```
python3 scripts/multitask_model_v01.py --device cuda
```
