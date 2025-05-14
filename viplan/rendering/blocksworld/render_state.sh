#!/bin/bash
# For triton usage
# module load libxi

# Default values for arguments
render_objects=""
root=""

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --render-state) render_state="$2"; shift ;;
        --root) root="$2"; shift ;;
        --output-dir) output_dir="$2"; shift ;;
        --use-gpu) use_gpu="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# If use_gpu is not set, default to 1 (use GPU)
if [ -z "$use_gpu" ]; then
    echo "Setting default value for use_gpu"
    use_gpu=1
fi

if [ "$use_gpu" == "0" ]; then
    use_gpu=0
elif [ "$use_gpu" == "1" ]; then
    use_gpu=1
else
    echo "Invalid value for use_gpu. It should be 0 or 1."
    exit 1
fi

echo "Renderer using gpu: $use_gpu"

cd $root

#blenderdir=$(echo blender-2.*/)
# blenderdir=$(echo blender-3.*/)
blenderdir=$(echo $root/blender-3.0.0-linux-x64)
$blenderdir/blender -noaudio --background --python $root/viplan/rendering/blocksworld/render.py -- \
      --output-dir "$output_dir"                          \
      --render-num-samples 512                          \
      --width 300                                       \
      --height 200                                      \
      --render-state "$render_state"                    \
      --base-scene-blendfile $root/data/blocksworld_rendering/base_scene.blend \
      --properties-json $root/data/blocksworld_rendering/properties.json \
      --shape-dir $root/data/blocksworld_rendering/shapes \
      --material-dir $root/data/blocksworld_rendering/materials \
      --use-gpu $use_gpu
      # --use-gpu 0 \