#!/bin/bash

# Exit if any command fails
set -e

# Download Blender 3.0.0
wget https://download.blender.org/release/Blender3.0/blender-3.0.0-linux-x64.tar.xz

# Extract the archive
tar xf blender-3.0.0-linux-x64.tar.xz

# Add current directory to Blender's Python site-packages
CLEVR_PATH=$(echo blender*/3.*/python/lib/python*/site-packages/)
echo $PWD > "${CLEVR_PATH}clevr.pth"

# Clean up
rm blender-3.0.0-linux-x64.tar.xz

echo "Blender setup for Blocksworld completed."
