Installation Recommendations

# We are about to set up a python virtual environment providing the dependencies for the maltose code. Choose the path for the virtual environment according to your preferences:
VENVPATH=$HOME/venvs/maltose-pip

# Set up a new python environment:
python3 -m venv $VENVPATH
# Activate the new environment:
source $VENVPATH/bin/activate

# Upgrade the pip package manager:
python3 -m pip install --upgrade pip

# Install the basic requirements (including torch)
python3 -m pip install -r requirements0.txt

# Install the main requirements (depending on torch)
python3 -m pip install -r requirements1.txt
