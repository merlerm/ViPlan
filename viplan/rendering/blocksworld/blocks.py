import json
import random
from datetime import datetime as dt

properties = {}


def load_colors(args):
    # Load the property file
    with open(args.properties_json, 'r') as f:
        properties.update(json.load(f))

        # changes color value range from 0-255 to 0-1
        properties["colors"] = [
            tuple(float(c) / 255.0 for c in rgb) + (1.0,)
            for rgb in properties['colors']
        ]

        if not args.randomize_colors:
            # extract exactly the same numbr of colors as the objects
            # from the top in the order as written in the json file
            properties["colors"] = properties["colors"][:args.num_objects]

    return


def initialize_parser_input_options(parser):
    # Input options
    parser.add_argument('--base-scene-blendfile', default='data/base_scene.blend',
                        help="Base blender file on which all scenes are based; includes " +
                        "ground plane, lights, and camera.")
    parser.add_argument('--properties-json', default='data/properties.json',
                        help="JSON file defining objects, materials, sizes, and colors. " +
                        "The \"colors\" field maps from CLEVR color names to RGB values; " +
                        "The \"sizes\" field maps from CLEVR size names to scalars used to " +
                        "rescale object models; the \"materials\" and \"shapes\" fields map " +
                        "from CLEVR material and shape names to .blend files in the " +
                        "--object-material-dir and --shape-dir directories respectively.")
    parser.add_argument('--shape-dir', default='data/shapes',
                        help="Directory where .blend files for object models are stored")
    parser.add_argument('--material-dir', default='data/materials',
                        help="Directory where .blend files for materials are stored")
    parser.add_argument('--randomize-colors', action="store_true",
                        help="Select the object color from all colors available in properties.json for each state."
                        + " If not present, the list of colors is truncated to match the number of objects"
                        + " during the initialization.")
    parser.add_argument('--allow-duplicates', action="store_true",
                        help="Allow duplicate objects")
    parser.add_argument('--seed', default=0, type=int,
                        help="Random seed")


def initialize_parser_environment_options(parser):
    # Settings for objects
    parser.add_argument('--num-objects', default=4, type=int,
                        help="The number of objects to place in each scene")

    parser.add_argument('--table-size', default=5, type=int,
                        help="The approximate table size relative to the large object size * 1.5.")

    parser.add_argument('--object-jitter', default=0.0, type=float,
                        help="The magnitude of random jitter to add to the x,y position of each block.")


def initialize_parser_output_options(parser, prefix):
    parser.add_argument('--filename-prefix', default=prefix,
                        help="This prefix will be prepended to the rendered images and JSON scenes")
    parser.add_argument('--output-dir', default='output',
                        help="The directory where output will be stored. It will be " +
                        "created if it does not exist.")
    parser.add_argument('--save-blendfiles', type=int, default=0,
                        help="Setting --save-blendfiles 1 will cause the blender scene file for " +
                        "each generated image to be stored in the directory specified by " +
                        "the --output-blend-dir flag. These files are not saved by default " +
                        "because they take up ~5-10MB each.")
    parser.add_argument('--version', default='1.0',
                        help="String to store in the \"version\" field of the generated JSON file")
    parser.add_argument('--license',
                        default="Creative Commons Attribution (CC-BY 4.0)",
                        help="String to store in the \"license\" field of the generated JSON file")
    parser.add_argument('--date', default=dt.today().strftime("%m/%d/%Y"),
                        help="String to store in the \"date\" field of the generated JSON file; " +
                        "defaults to today's date")


def initialize_parser_rendering_options(parser):
    # Rendering options
    parser.add_argument('--use-gpu', default=0, type=int,
                        help="Setting --use-gpu 1 enables GPU-accelerated rendering using CUDA. " +
                        "You must have an NVIDIA GPU with the CUDA toolkit installed for " +
                        "to work.")
    parser.add_argument('--width', default=320, type=int,
                        help="The width (in pixels) for the rendered images")
    parser.add_argument('--height', default=240, type=int,
                        help="The height (in pixels) for the rendered images")
    parser.add_argument('--key-light-jitter', default=0.0, type=float,
                        help="The magnitude of random jitter to add to the key light position.")
    parser.add_argument('--fill-light-jitter', default=0.0, type=float,
                        help="The magnitude of random jitter to add to the fill light position.")
    parser.add_argument('--back-light-jitter', default=0.0, type=float,
                        help="The magnitude of random jitter to add to the back light position.")
    parser.add_argument('--camera-jitter', default=0.0, type=float,
                        help="The magnitude of random jitter to add to the camera position")
    parser.add_argument('--render-num-samples', default=512, type=int,
                        help="The number of samples to use when rendering. Larger values will " +
                        "result in nicer images but will cause rendering to take longer.")
    parser.add_argument('--render-min-bounces', default=8, type=int,
                        help="The minimum number of bounces to use for rendering.")
    parser.add_argument('--render-max-bounces', default=8, type=int,
                        help="The maximum number of bounces to use for rendering.")
    parser.add_argument('--render-tile-size', default=256, type=int,
                        help="The tile size to use for rendering. This should not affect the " +
                        "quality of the rendered image but may affect the speed; CPU-based " +
                        "rendering may achieve better performance using smaller tile sizes " +
                        "while larger tile sizes may be optimal for GPU-based rendering.")


def random_dict(dict):
    return random.choice(list(dict.items()))


def dump(obj):
    def rec(obj):
        if hasattr(obj, "__dict__"):
            res = {
                k: rec(v)
                for k, v in vars(obj).items()
            }
            res["__class__"] = obj.__class__.__name__
            return res
        elif isinstance(obj, dict):
            return {k: rec(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return list(rec(v) for v in obj)
        elif isinstance(obj, tuple):
            return list(rec(v) for v in obj)
        else:
            return obj
    return rec(obj)

# a = A() ; a.b=1 ; a.c="2" ; a.d=1.1 ; a.e=A() ; a.e.b=3 ; a.f = [2, A()] ; a.g = {"a": A()} ; a.h = (2, A()) ;
# dump(a)
# dump(undump(dump(a)))


def undump(obj):
    def rec(obj):
        if isinstance(obj, dict):
            if "__class__" in obj:
                cls = eval(obj["__class__"])
                res = cls.__new__(cls)
                for k, v in obj.items():
                    if k != "__class__":
                        vars(res)[k] = rec(v)
                return res
            else:
                return {k: rec(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return list(rec(v) for v in obj)
        else:
            return obj
    return rec(obj)


class Unstackable(Exception):
    pass

# TODO hardcoded for now, do we need to put this in properties.json?
from viplan.planning.conversion import block_id, block_letter
color_map = {
    block_id['R']: [1, 0, 0, 1],
    block_id['G']: [0, 0.8, 0, 1],
    block_id['B']: [0, 0, 0.8, 1],
    block_id['Y']: [1, 1, 0, 1],
    block_id['P']: [0.2, 0, 0.5, 1],
    block_id['O']: [1, 0.5, 0, 1],
}


class Block(object):
    def __init__(self, idx, **kwargs):

        # Leaving for now as only shape is cube anyways
        shape_name, self.shape = random_dict(properties['shapes'])
        if 'color' in kwargs:
            self.color = kwargs['color']
        else:
            self.color = random.choice(properties['colors'])
        _, self.size = random_dict(properties['sizes'])
        _, self.material = random_dict(properties['materials'])
        self.rotation = 360.0 * random.random()
        self.stackable = properties['stackable'][shape_name] == 1
        self.location = [0, 0, 0]
        self.id = idx
        pass

    @property
    def x(self):
        return self.location[0]

    @property
    def y(self):
        return self.location[1]

    @property
    def z(self):
        return self.location[2]

    @x.setter
    def x(self, newvalue):
        self.location[0] = newvalue

    @y.setter
    def y(self, newvalue):
        self.location[1] = newvalue

    @z.setter
    def z(self, newvalue):
        self.location[2] = newvalue

    def __eq__(o1, o2):
        if o1 is None:
            return False
        if o2 is None:
            return False
        return o1.id == o2.id

    def similar(o1, o2):
        if o1 is None:
            return False
        if o2 is None:
            return False
        return \
            o1.color == o2.color and \
            o1.size == o2.size and \
            o1.material == o2.material

    def overlap(o1, o2):
        return (abs(o1.x - o2.x) < (o1.size + o2.size))

    def stable_on(o1, o2):
        return (abs(o1.x - o2.x) < o2.size)

    def above(o1, o2):
        return o1.overlap(o2) and (o1.z > o2.z)


class State(object):
    "Randomly select a list of objects while avoiding duplicates"

    def __init__(self, matrix, properties_json='./data/properties.json', object_jitter=0.01, seed=None):
        if seed is not None:
            random.seed(seed)
            
        objects = []
        
        with open(properties_json, 'r') as f:
            properties.update(json.load(f))

        # Create a state with a fixed set of objects
        self.table_size = len(matrix)
        self.max_column_height = len(matrix[0])
        self.object_jitter = object_jitter 

        unit = max(properties['sizes'].values())
        base_padding = 1  # Base padding value
        padding = base_padding * max(0, (5 - self.table_size) / 5)  # Adjust padding based on the number of columns
        min_x = -6 + padding
        max_x = 6 - padding  # Hardcoded after trial and error with added padding

        # Compute a dictionary returning the x position for every column, evenly spread out from min_x to max_x
        if self.table_size > 1:
            spacing = (max_x - min_x) / (self.table_size - 1)
        else:
            spacing = 0
        self.column_x = {
            i: min_x + i * spacing
            for i in range(self.table_size)
        }

        # Create the objects
        objects = []
        for i in range(self.table_size):
            for j in range(self.max_column_height):
                if matrix[i][j] == 0:
                    continue
                o = Block(block_letter[int(matrix[i][j])], color=color_map[matrix[i][j]])
                o.x = self.column_x[i]
                o.z = o.size + j * o.size * 2
                # print(
                #     f"Created object with id {i} color {o.color} at position {o.x}, {o.z} and size {o.size}")
                objects.append(o)

        self.objects = objects
        # print("Successfully created a state with ", len(objects), " objects.")
        pass

    def for_rendering(self):
        return [vars(o) for o in sorted(self.objects, key=(lambda o: o.id))]

    def dump(self):
        return dump(self)

    @staticmethod
    def undump(data):
        return undump(data)

    def shuffle(self):
        """destructively modify the list of objects using shuffle1."""
        objs = self.objects.copy()
        self.objects.clear()
        for oi in objs:
            self.shuffle1(oi)
            self.objects.append(oi)

    def shuffle1(self, oi, force_change=False):
        """destructively modify an object by choosing a random x position and put it on top of existing objects.
     oi itself is not inserted to the list of objects."""
        # note: if a cube is rotated by 45degree, it should consume 1.41 times the size
        unit = max(properties['sizes'].values())
        max_x = unit * 2 * self.table_size

        if force_change:
            object_below = self.object_just_below(oi)

        trial = 0
        fail = True
        while fail and trial < 100:
            fail = False
            oi.x = max_x * \
                ((random.randint(0, self.table_size-1) / (self.table_size-1)) - 1/2)
            oi.z = 0
            for oj in self.objects:
                if oi.overlap(oj):
                    if not oj.stackable:
                        fail = True
                        break
                    if not oi.stable_on(oj):
                        fail = True
                        break
                    oi.z = max(oi.z, oj.z + oj.size)
            oi.z += oi.size
            if force_change:
                new_object_below = self.object_just_below(oi)
                if object_below == new_object_below:
                    # is not shuffled!
                    fail = True
            trial += 1

        if fail:
            raise Unstackable("this state is not stackable")
        pass

    def wiggle(self):
        """wiggles all objects by adding a jitter to the x coordinate of the objects"""
        unit = max(properties['sizes'].values())
        for oi in self.objects:
            oi.x += random.gauss(0.0, self.object_jitter * unit)
            
    def wiggle_specific(self, o):
        """wiggles a specific object by adding a jitter to the x coordinate of the object and its rotation"""
        unit = max(properties['sizes'].values())
        o.x += random.gauss(0.0, self.object_jitter * unit)
        o.rotation += random.gauss(0.0, 1.0)

    def tops(self):
        """returns a list of objects on which nothing is on top of, i.e., it is the top object of the tower."""
        tops = []
        for o1 in self.objects:
            top = True
            for o2 in self.objects:
                if o1 == o2:
                    continue
                if o2.above(o1):
                    top = False
                    break
            if top:
                tops.append(o1)
        return tops

    def objects_below(self, o):
        results = []
        for other in self.objects:
            if o != other and o.above(other):
                results.append(other)
        return results

    def object_just_below(self, o):
        objects_below = self.objects_below(o)
        if len(objects_below) == 0:
            return None
        else:
            result = objects_below[0]
            for other in objects_below[1:]:
                if result.z < other.z:
                    result = other
            return result

    def random_action(self):
        method = random.choice([self.action_move])
        method()
        # storing the name of the action. This is visible in the json file
        self.last_action = method.__name__
        pass

    def action_move_specific(self, o1, o2):
        # Move o1 to the top of o2 if possible
        if o1 == o2:
            return False
        if o1.above(o2):
            print("o1 is already above o2")
            return False
        if not o2.stackable:
            print("o2 is not stackable")
            return False
        for o in self.objects:
            if o == o1 or o == o2:
                continue
            if o.above(o1):
                print("o1 is not the top object and thus cannot be moved")
                return False
            if o.above(o2):
                print("o2 has other objects above it and is not a valid destination")
                return False

        o1.x = o2.x
        o1.z = o2.z + o2.size * 2
        self.wiggle_specific(o1)
        self.wiggle_specific(o2)
        self.last_action = self.action_move_specific.__name__
        return True
    
    def action_move_column(self, o, column_id):
        # Move o to an empty column
        if o.x == self.column_x[column_id]:
            print("o is already in the column")
            return False
        if not o.stackable:
            print("o is not stackable")
            return False
        for o1 in self.objects:
            if o == o1:
                continue
            if o1.above(o):
                print("o is not the top object and thus cannot be moved")
                return False
            if o1.x == self.column_x[column_id]:
                print("column has other objects above it and is not a valid destination")
                return False

        o.x = self.column_x[column_id]
        o.z = o.size # put it on the ground -> if there are blocks use action_move_specific instead
        self.wiggle_specific(o)
        self.last_action = self.action_move_column.__name__
        return True

    def action_move(self):
        o = random.choice(self.tops())
        index = self.objects.index(o)
        self.objects.remove(o)
        self.shuffle1(o, force_change=True)
        self.objects.insert(index, o)
        # note: do not change the order of the object.
        pass

    def action_change_material(self):
        o = random.choice(self.tops())
        tmp = list(properties['materials'].values())
        tmp.remove(o.material)
        o.material = random.choice(tmp)
        pass
    
    def get_block(self, block_id):
        for o in self.objects:
            if o.id == block_id:
                return o
        return None