# From https://doi.org/10.14459/2019mp1507656
for FILE in \
  README \
  atomic_energies.ods \
  df_62k.json \
  df_31k.json \
  df_5k.json \
  SHA512sums # \
  # tutorial.ipynb \
  # helpers.py \
do
  if [ ! -e "$FILE" ]
  then
    echo "Downloading $FILE"
    RSYNC_PASSWORD=m1507656 rsync -az rsync://m1507656@dataserv.ub.tum.de/m1507656/$FILE ./
  else
    echo "$FILE exists already!"
  fi
done
