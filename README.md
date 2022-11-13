# MaLTOSe - Machine Learning Tailoring Organic Semiconductors
-- Machine-learning code for predicting molecular properties --

## Installation from source

### Option 1: Using a python virtual environment created with venv and pip

Choose the path for the virtual environment according to your preferences:

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
The requirements.txt contains at least one package (torch-scatter) that depends on torch already being installed. Therefore, if you do not have torch installed yet, install the basic requirement (including torch, but not torch-scatter) first:

```
python3 -m pip install -r requirements0.txt
```

Then, installed all the other requirements:

```
python3 -m pip install -r requirements.txt
```

Build the maltose package:

```
python3 -m build
```

Install the maltose package:

```
pip install dist/maltose-0.0.1-py3-none-any.whl
```

You're ready to go!

### Option 2: Using a conda virtual environment

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


## The maltose library

Now, the contents of the library src/maltose can be imported and used in python.
The package contains:

- Some helper functions for format conversions of molecular data (maltose.conversions).
- Some helpers for preparing the primary data (maltose.primary_data).
- The Set2Set output module (in maltose.output_modules).
- The functionality for multitask training (maltose.atoms.MultitaskAtomsData, maltose.loss.build_gated_mse_loss, maltose.metrics.MultitaskMetricWrapper).


## Running experiments

All experiments are executed via python scripts in the ./scripts folder. These scripts write to and read from the following directories, by default placed next to the sripts directory:

- scratch/ directroy that is used only during the preparation of the training/validation/test data. Can be deleted afterwards.
- data/ is the data directory that is written during the preparation of the training/validation/test data.
- models/ is the diectory for saving trained models. Also some derived data, like accuracy measures of the model on the test sets, are saved here.
- figures/ folder is written to by skripts/make_plots.py.

Remark on file locations: Some scripts accept command-line parameters to specify these folder locations. If all scripts are called from the same parent directory of the scripts folder, then the default locations should be fine.

### To prepare the training data, execute:

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

### To train a maltose model, run:

```
python3 scripts/train_model.py --config <conig_name>
```
where, for example, conig_name = "multitask_model_v01".

Note that this may take some days or even weeks, depending on your hardware. If available, run the training on a GPU:

```
python3 scripts/train_model.py --config <conig_name> --device cuda
```

### For evaluation, run:

```
python3 scripts/compute_errors.py --config <conig_name> --device <device>
```
and:

```
python3 scripts/make_plots.py --config <conig_name> --device <device>
```
