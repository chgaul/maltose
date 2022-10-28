# To download and prepare al primary data needed in this project,
# do the following:

python3 alchemy_to_schnet.py
sh oe62_0_download.sh
python3 oe62_1_to_schnet.py
python3 hopv_to_schnet.py
python3 kuzmich_to_xyz.py

# The subfolder "scratch" may be deleted afterwards.
