#!/bin/bash
# Define source file and destination directory
SOURCE_FILES=(
  "/home/hana/Documents/dalhousie/training/CREATE/Data_Module/codes/FFT-analysis/bbm_tide_specs.pdf"
  "/home/hana/Documents/dalhousie/training/CREATE/Data_Module/codes/FFT-analysis/adcp_tidal_analysis.pdf"
  "/home/hana/Documents/dalhousie/training/CREATE/Data_Module/codes/eof-analysis/bbm_space_eof_hana.pdf"
  "/home/hana/Documents/dalhousie/training/CREATE/Data_Module/codes/eof-analysis/eof_ceof_comp_bbm_hana.pdf"
  "/home/hana/Documents/dalhousie/training/CREATE/Data_Module/codes/eof-analysis/eof_error_map_adcp.pdf"
  "/home/hana/Documents/dalhousie/training/CREATE/Data_Module/codes/linalg/linalg_background_hana.pdf"
  "/home/hana/Documents/dalhousie/training/CREATE/Data_Module/codes/wavelets/wavelet_cwt_tutorial.pdf"
)

DEST_DIR="/home/hana/Documents/dalhousie/training/CREATE/Data_Module/codes/lab_book"

# Create the destination directory if it doesn't exist (optional but recommended)
# mkdir -p "$DEST_DIR"

# Copy the file
for f in "${SOURCE_FILES[@]}"; do
  cp "$f" "$DEST_DIR/"
done
