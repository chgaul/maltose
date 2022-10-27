# From https://doi.org/10.14459/2019mp1507656
for FILE in \
  README \
  df_62k.json \
  SHA512sums # \
  # atomic_energies.ods \
  # df_31k.json \
  # df_5k.json \
  # tutorial.ipynb \
  # helpers.py \
do
  if [ ! -e "scratch/OE62/$FILE" ]
  then
    echo "Downloading $FILE"
    mkdir -p scratch/OE62/
    RSYNC_PASSWORD=m1507656 rsync -az rsync://m1507656@dataserv.ub.tum.de/m1507656/$FILE scratch/OE62/
  else
    echo "$FILE exists already!"
  fi
done
