# MaLTOSe - Machine Learning Tailoring Organic Semiconductors
-- Machine-learning code for predicting molecular properties --

## Installation from source

### [Optional] We recommend setting up a python virtual environment providing the dependencies for the maltose code. Choose the path for the virtual environment according to your preferences:

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

### Install requirements
The requirements.txt contains at least one package (torch-scatter) depends on torch already being installed. Therefore, if you do not have torch installed yet, install the basic requirement (including torch, but not torch-scatter) first:

```
python3 -m pip install -r requirements_0.txt
```

Then, make sure all the other requirements are installed, too:

```
python3 -m pip install -r requirements.txt
```

Install the maltose package

```
pip install .
```

You're ready to go!