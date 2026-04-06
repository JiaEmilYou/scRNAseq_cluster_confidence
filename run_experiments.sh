#!/usr/bin/env bash

set -e

for dataset in pbmc3k pbmc68k_reduced paul15
do
  for res in 0.5 1.0 1.5 2.0 2.5
  do
    name="${dataset}_res_${res}"
    echo "Running: $name"

    dvc exp run -f -n $name \
      -S scanpy.dataset_name=$dataset \
      -S scanpy.leiden_resolution=$res
  done
done