#!/bin/bash

# File to store logs
output_file="pdi_gen.log"
output_folder="generated_pdi_insts"

mkdir -p "$output_folder"

#squre
gen_sizes=(
  "256 1280 3584"
  "256 1280 1024"
  "256 1280 4096"
  "256 4352 1024"

#   "768 2048 2560"
#   "768 2048 2048"
#   "768 2048 8192"

#   "768 16384 2048"
)

TILE="64x128x64"
COLS="8c"
DT="bf16_bf16"

mkdir -p "$DATA_DIR"
for size in "${gen_sizes[@]}"; do
    read M K N <<< "$size"

    # Create subdirectory for this size
    size_dir="$output_folder/${M}x${K}x${N}"
    mkdir -p "$size_dir"

    echo "Running with M=${M}, K=${K}, N=${N}"

    # make -f Makefile.chess devicename=npu2 M=$M K=$K N=$N | tee -a $output_file
    make devicename=npu2 M=$M K=$K N=$N | tee -a $output_file
    xclbinutil --dump-section AIE_PARTITION:JSON:aie_partition.json --force --input build/final_${M}x${K}x${N}_${TILE}_${COLS}.xclbin

    latest_pdi=$(ls -t *.pdi 2>/dev/null | head -n 1)
    if [[ -z "$latest_pdi" ]]; then
        echo "No .pdi files found in the current directory."
        return 1
    fi
    echo "generated PDI" $latest_pdi
    mv "$latest_pdi" "$size_dir/final_${M}x${K}x${N}_${TILE}_${COLS}_${DT}.pdi"
    cp "build/insts_${M}x${K}x${N}_${TILE}_${COLS}.txt" "$size_dir/insts_${M}x${K}x${N}_${TILE}_${COLS}_${DT}.txt"
done