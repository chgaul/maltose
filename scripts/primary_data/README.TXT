# To download and prepare al primary data needed in this project,
# run the following:

cd ../../
python3 scripts/primary_data/prepare_qm9.py
python3 scripts/primary_data/alchemy_to_schnet.py
python3 scripts/primary_data/prepare_oe62.py
python3 scripts/primary_data/hopv_to_schnet.py
python3 scripts/primary_data/kuzmich_to_xyz.py

# Results will be produced in the folder "data".
# The folder "scratch" may be deleted afterwards.
