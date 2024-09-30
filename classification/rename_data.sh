#!/bin/bash

# Navigate to the directory containing the JPEG files
cd /joe/data/ImageNet-Zip/val

# Loop through all JPEG files
for file in ILSVRC2012_val_*.JPEG; do
  # Extract the part before the first underscore and the part after the last underscore
  new_name=$(echo "$file" | sed -E 's/(ILSVRC2012_val_[0-9]+)_.+(\.JPEG)/\1\2/')
  
  # Rename the file
  mv "$file" "$new_name"
done