import sys
import argparse
import json
import os
import pickle
import numpy as np
import random

INSIDE_BLENDER = True
try:
    import bpy
    import bpy_extras
    from mathutils import Vector
except ImportError as e:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import viplan.rendering.blocksworld.utils as utils
        import viplan.rendering.blocksworld.blocks as blocks
        from viplan.rendering.blocksworld.blocks import State, Unstackable, load_colors
        from viplan.rendering.blocksworld.render_utils import render_scene
    except ImportError as e:
        print(e)
        # Print stack trace
        import traceback
        traceback.print_exc()
        print("\nERROR")
        print("Running render.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.5/site-packages/clevr.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.78).")
        sys.exit(1)


def initialize_parser():
    parser = argparse.ArgumentParser()

    # Input options
    blocks.initialize_parser_input_options(parser)

    # Environment options
    blocks.initialize_parser_environment_options(parser)

    # Output settings
    blocks.initialize_parser_output_options(parser, prefix='CLEVR')

    parser.add_argument('--num-samples-per-state', default=1, type=int,
                        help="The number of images to render per logical states")
    parser.add_argument('--render-state', type=str, default=None,
                        help="Path to a pickle file containing the objects to render coming from the simulator")

    # Rendering options
    blocks.initialize_parser_rendering_options(parser)

    return parser


def path(dir, i, presuc, j, ext):
    if isinstance(i, int):
        i = "{:06d}".format(i)
    if isinstance(j, int):
        j = "{:03d}".format(j)
    return os.path.join(args.output_dir, dir, "_".join(["CLEVR", i, presuc, j])+"."+ext)


def main(args):
    import copy

    print(f"Loading render state from {args.render_state}")
    if not os.path.exists(args.render_state):
        print(f"Error: render state file {args.render_state} not found.")
        sys.exit(1)
    state = pickle.load(open(args.render_state, "rb"))
    
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    render_scene(args, output_image=os.path.join(args.output_dir, "render.png"), output_scene=os.path.join(args.output_dir, "scene.json"), objects=state.for_rendering())


if __name__ == '__main__':
    parser = initialize_parser()
    if INSIDE_BLENDER:
        # Run normally
        argv = utils.extract_args()
        args = parser.parse_args(argv)
        print(args)
        main(args)
    elif '--help' in sys.argv or '-h' in sys.argv:
        parser.print_help()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render.py --help')
