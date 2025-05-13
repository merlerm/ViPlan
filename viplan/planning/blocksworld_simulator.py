import io
import os
import time
import copy
import pickle
import random
import datetime
import subprocess
import numpy as np
import unified_planning
import matplotlib.pyplot as plt

from PIL import Image
from collections import defaultdict
from viplan.code_helpers import get_logger
from viplan.planning.conversion import predicates_to_numpy, state_to_bird, _sort_columns
from viplan.rendering.blocksworld.blocks import State

from unified_planning.model.object import Object
from unified_planning.plans.plan import ActionInstance

from fasteners import InterProcessLock

class BlocksworldSimulator():
    def __init__(self, problem: unified_planning.model.Problem, 
                 root_path: os.PathLike,
                 fail_probability: float = 0.0,
                 render_path: os.PathLike = None,
                 logger=get_logger('info'),
                 seed=None,
                 use_gpu_rendering=True):
        
        self.logger = logger
        # Random is used within the render_state to wiggle the rotation of the blocks
        self.seed = seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.problem = problem
        self.root_path = root_path
        #! Render path must be absolute
        # Render path is used to save the render for the current state (temporary)
        if render_path is None:
            unique_id = os.getenv('SLURM_JOB_ID', None) # Unique ID avoids two jobs overwriting each other
            if unique_id is not None:
                array_id = os.getenv('SLURM_ARRAY_TASK_ID', None)
                if array_id is not None:
                    unique_id += f"_{array_id}"
            if unique_id is None:
                unique_id = str(datetime.datetime.now().timestamp()).replace('.', '_')
            else:
                unique_id += str(datetime.datetime.now().timestamp()).replace('.', '_')
            self.render_path = os.path.join(self.root_path, f'render_files_{unique_id}')
        else:
            self.render_path = render_path
        self.render_script = os.path.join(self.root_path, 'viplan', 'rendering', 'blocksworld', 'render_state.sh')
        self.use_gpu_rendering = "1" if use_gpu_rendering else "0"
        self.logger.info(f"GPU rendering: {use_gpu_rendering}")
        self.fail_probability = fail_probability
        
        self.fail_noop_p = 0.5 # probability of doing nothing
        self.fail_drop_goal_p = 0.25 # probability of dropping the block near the goal (the remaining probability is dropping it near the start)
                
        self._init_state_from_problem()    
        logger.debug(f"Initial state: {self.state}")
        
        # TODO add path for properties.json as an argument?
        self.properties_json = os.path.join(self.root_path, 'data', 'blocksworld_rendering', 'properties.json')
        self.render_state = State(list(self.np_state[0:5]), properties_json=self.properties_json, seed=seed)
        
    def __del__(self):
        if os.path.exists(self.render_path):
            for filename in os.listdir(self.render_path):
                os.remove(os.path.join(self.render_path, filename))
            os.rmdir(self.render_path)
        self.logger.info(f"Deleted render path {self.render_path}")
        
    @property
    def np_state(self):
        return predicates_to_numpy(self.state)
    @property
    def bird_str(self):
        return state_to_bird(self.np_state)
    
    @property
    def goal_fluents(self):
        goal_predicates = self.problem.goals[0] # Assuming there are just a set of "ands" in the goal, does not support single predicate goals or negations (as they never happen in the dataset)
        # TODO move over logic from the igibson client that supports negations
        assert goal_predicates.is_and()
        for goal in goal_predicates.args:
            assert goal.is_fluent_exp(), "Goal structure was not a simple conjunction of fluents"
        goal_fluents = goal_predicates.args
        return goal_fluents
    
    @property
    def goal_reached(self):
        goal_fluents = self.goal_fluents
        for fluent in goal_fluents:
            fluent_name = fluent.fluent().name
            fluent_args = [a.object().name for a in fluent.args]
            if not self.state[fluent_name][','.join(fluent_args)]:
                self.logger.debug(f"Goal fluent {fluent_name} with args {fluent_args} not satisfied")
                return False
        
        self.logger.info("Goal reached")
        return True
            
    
    def _init_state_from_problem(self):
        self.all_objects = {str(self.problem.user_types[type_]): list(self.problem.objects(self.problem.user_types[type_])) for type_ in range(len(self.problem.user_types))}
        self.state = {}
        # Initialize state with default values
        for fluent in self.problem.fluents:
            fluent_name = fluent.name
            arg_names = [arg.name for arg in fluent.signature]
            default_value = self.problem.fluents_defaults[fluent]
            fluent_dict = defaultdict(lambda *arg_names: default_value.is_true())
            self.state[fluent_name] = fluent_dict

        # Fill in values explicitly defined in the problem
        for init, value in self.problem.initial_values.items():
            fluent = init.fluent()
            fluent_name = fluent.name
            arg_names = [str(arg) for arg in init.args]
            self.state[fluent_name][','.join(arg_names)] = value.is_true()
            
        self.column_order = _sort_columns(self.state)
        
        # Update rightof and leftof relationships
        self.state = self._update_indirect_relationships(self.state)
    
    def __str__(self):
        np_arr = predicates_to_numpy(self.state, use_letters=True)
        np_arr = np.flip(np_arr, axis=1).T
        return str(np_arr)
    
    import io

    def _label_columns(self, image, column_labels=None, add_lines=True):
        
        if column_labels is None:
            column_labels = self.column_order
        
        fig, ax = plt.subplots(figsize=(image.width / 100, image.height / 100), dpi=100)
        ax.imshow(image)
        width = ax.get_xlim()[1]
        
        # padding = width / (4 * len(column_labels))
        padding = 0
        effective_width = width - padding  # only subtract padding at the beginning
        
        if add_lines:
            cell_width = effective_width / len(column_labels)
            # Draw vertical lines at column boundaries: first line at start and then at each boundary
            for i in range(len(column_labels) + 1):
                x_line = padding + i * cell_width
                ax.axvline(x=x_line, color='black', linestyle='--', linewidth=0.75)
        
        cell_width = effective_width / len(column_labels)
        for i, label in enumerate(column_labels):
            x = padding + (i + 0.5) * cell_width
            y = ax.get_ylim()[0] - 10
            ax.text(x, y, label, fontsize=10, ha='center', va='center', color='black')
            
        plt.axis('off')
        plt.tight_layout()
        # remove all white space and only keep the image
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        plt.close()
        return img
    
    def render(self, label_columns=True):
        os.makedirs(self.render_path, exist_ok=True)
        render_state_path = os.path.join(self.render_path, "render_state.pkl")
        # os.remove(render_state_path) if os.path.exists(render_state_path) else None
        
        lock = InterProcessLock(render_state_path)
        with lock:
            with open(render_state_path, 'wb') as f:
                pickle.dump(self.render_state, f)
            self.logger.debug(f"Render state saved to temporary file: {render_state_path}")
            
            start = time.perf_counter()
            process = subprocess.run(['bash', self.render_script, '--root', self.root_path, '--render-state', f.name, '--output-dir', self.render_path, '--use-gpu', self.use_gpu_rendering], check=True, stdout=subprocess.PIPE)
            self.logger.info(f"Rendered state in {time.perf_counter() - start:.2f} seconds")
            self.logger.debug(f"Render process finished with return code {process.returncode}")
            os.remove(render_state_path)
        
            render_image_path = os.path.join(self.render_path, "render.png")
            if os.path.exists(render_image_path):
                render_image = Image.open(render_image_path)
                os.remove(render_image_path)
                os.remove(os.path.join(self.render_path, "scene.json")) if os.path.exists(os.path.join(self.render_path, "scene.json")) else None
                os.rmdir(self.render_path)
            
                if label_columns:
                    render_image = self._label_columns(render_image)
            
                return render_image
            
            else:
                self.logger.error(f"Render image not found at {render_image_path}")
                os.remove(os.path.join(self.render_path, "scene.json")) if os.path.exists(os.path.join(self.render_path, "scene.json")) else None
                os.rmdir(self.render_path)
                return None
        
    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
            random.seed(seed)
            np.random.seed(seed)
        
        self.logger.info(f"Resetting simulator with seed {self.seed}")
        self._init_state_from_problem()
        self.render_state = State(list(self.np_state[0:5]), properties_json=self.properties_json, seed=self.seed)
        
    def set_state(self, new_state):
        self.state = new_state
        self.render_state = State(list(self.np_state[0:5]), properties_json=self.properties_json, seed=self.seed)
    
    # Helper functions
    
    def _get_column_blocks(self, column):
        blocks = []
        inColumn = self.state['incolumn']
        for args, value in inColumn.items():
            if value and args.split(',')[1] == column:
                blocks.append(args.split(',')[0])
        return blocks
        
    def _top_block(self, column):
        blocks = self._get_column_blocks(column)
        clear = self.state['clear']
        for block in blocks:
            if clear[block]:
                return block
        return None
    
    def _get_column_idx(self, col):
        return self.column_order.index(col)
    
    # Needed to update leftof and rightof that are not immediately adjacent
    def _update_indirect_relationships(self, state):
        leftof = state['leftof']
        rightof = state['rightof']
        
        # First, collect all unique columns
        columns = set()
        for key in list(leftof.keys()) + list(rightof.keys()):
            c1, c2 = key.split(',')
            columns.update([c1, c2])
        
        # Build a graph from the immediate relationships.
        # For each key "c1,c2" with value True in rightof, we say that c1 is immediately to the right of c2.
        # Thus, we add an edge from c2 to c1.
        graph = {c: set() for c in columns}
        for key, value in rightof.items():
            if value:
                c1, c2 = key.split(',')
                graph[c2].add(c1)
        
        # For each column, use DFS to compute all columns that are to its right (directly or indirectly)
        def get_all_right_neighbors(col, visited=None):
            if visited is None:
                visited = set()
            for neighbor in graph[col]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    get_all_right_neighbors(neighbor, visited)
            return visited

        # Update rightof and leftof with indirect relationships
        for c in columns:
            # all columns to the right of c (directly or indirectly)
            all_right = get_all_right_neighbors(c)
            for r in all_right:
                key = f"{r},{c}"
                if not rightof.get(key, False):
                    rightof[key] = True
                    # Also update the inverse: if r is right of c, then c is left of r.
                    leftof[f"{c},{r}"] = True
                    
        return state
    
    # Recursively check the truth value of a node, which can contain either a fluent (predicate with arguments) or a logical operator
    # The state argument is needed because when applying effects the truth value of a fluent needs to be checked from the state BEFORE the effect was applied, since self.state will have incomplete updates
    def _check_value(self, node, grounded_args, state=None):
        if state is None:
            state = self.state
        if node.is_and():
            return all([self._check_value(arg, grounded_args, state) for arg in node.args])
        elif node.is_or():
            return any([self._check_value(arg, grounded_args, state) for arg in node.args])
        elif node.is_not():
            return not self._check_value(node.args[0], grounded_args, state)
        elif node.is_equals():
            arg1, arg2 = node.args
            if arg1.is_parameter_exp() or arg1.is_variable_exp() and arg2.is_parameter_exp() or arg2.is_variable_exp():
                return grounded_args[str(arg1)] == grounded_args[str(arg2)]
            else:
                return self._check_value(arg1, grounded_args, state) == self._check_value(arg2, grounded_args, state) 
        elif node.is_fluent_exp():
            fluent_name = node.fluent().name
            arg_names = [str(arg) for arg in node.args]
            args = [grounded_args[arg] for arg in arg_names]
            value = state[fluent_name][','.join(args)]
            return value if isinstance(value, bool) else value.is_true()
        else:
            raise ValueError("Unknown node type", node)

    # Apply a fluent effect to the state -> check the grounded arguments and update the truth value of the fluent
    def _apply_fluent(self, effect, fluent, grounded_args, state):
        arg_names = [str(arg) for arg in effect.fluent.args]
        args = [grounded_args[arg] for arg in arg_names]
        value = effect.value.is_true()
        # self.logger.debug("Setting fluent", fluent.name, "with args", args, "to", value)
        state[fluent.name][','.join(args)] = value

    # Apply an effect to the state, which can be either a simple fluent change, a conditional effect, a forall effect etc.
    def _apply_effect(self, effect, grounded_args, state):
        if effect.is_forall():
            assert len(effect.forall) == 1, "Only single forall supported"
            var = effect.forall[0]
            for value in self.all_objects[str(var.type)]:
                grounded_args[var.name] = str(value)
                if effect.is_conditional():
                    condition = effect.condition
                    # self.logger.debug(f"Checking condition {condition} with args {grounded_args}")
                    # self.logger.debug(f"Result: {self._check_value(condition, grounded_args, state)}")
                    if not self._check_value(condition, grounded_args, state):
                        continue
                self._apply_fluent(effect, effect.fluent.fluent(), grounded_args, state)
        elif effect.is_conditional():
            condition = effect.condition
            if self._check_value(condition, grounded_args, state):
                self._apply_fluent(effect, effect.fluent.fluent(), grounded_args, state)
        else:
            self._apply_fluent(effect, effect.fluent.fluent(), grounded_args, state)
    
    # Update truth values in the state dictionary, but only if they have changed during the current action
    # Initial state is the state before the action, effect state is the intermediate state after the current effect (but potentially not the final effect)
    def _update_state_dict(self, initial_state, effect_state):
        for fluent_name, fluent_dict in initial_state.items():
            for args, value in fluent_dict.items():
                if initial_state[fluent_name][args] != effect_state[fluent_name][args]:
                    self.state[fluent_name][args] = effect_state[fluent_name][args]
                    self.logger.debug("Fluent", fluent_name, "with args", args, "changed from", initial_state[fluent_name][args], "to", effect_state[fluent_name][args])
    
    # Actions
    
    def _get_specific_action(self, block: str, column: str, action_name: str = 'moveblock'):
        assert block in [ str(b) for b in self.all_objects['block'] ], f"Block {block} not found in all objects"
        assert column in [ str(c) for c in self.all_objects['column'] ], f"Column {column} not found in all objects"
        
        block_obj = Object(block, self.problem.user_type('block'))
        column_obj = Object(column, self.problem.user_type('column'))
        action = self.problem.action(action_name)
        
        action_instance = ActionInstance(action, [block_obj, column_obj])
        return action_instance

    def fail_action(self, action):
        self.logger.info(f"Action {action} fails.")
        # Three possible outcomes:
        # 1. Do nothing - simulates a failed grasp. Do this with probability 0.5
        # 2. Move the block to one of the columns that are near the goal - simulates a failed move. Do this with probability 0.25
        # 3. Move the block to one of the columns that are near the start - simulates a failed move. Do this with probability 0.25
        
        rand = random.random()
        
        # 1. Do nothing
        if rand < self.fail_noop_p:
            self.logger.info(f"Selected outcome: Do nothing")
            return None
        
        else:
            columns = self.column_order
            block = action.actual_parameters[0].object().name
            
            # 2. Move the block to one of the columns that are near the goal
            if rand < self.fail_noop_p + self.fail_drop_goal_p:
                self.logger.info(f"Selected outcome: Move block {block} to a column near the goal")
                col_idx = self._get_column_idx(action.actual_parameters[1].object().name)
            
            # 3. Move the block to one of the columns that are near the start
            else:
                self.logger.info(f"Selected outcome: Move block {block} to a column near the start")
                # Find the starting column using the "incolumn" fluent
                incolumn = self.state['incolumn']
                for args, value in incolumn.items():
                    if value and args.split(',')[0] == block:
                        current_column = args.split(',')[1]
                        break
                if current_column is None:
                    self.logger.error(f"Block {block} not found in any column")
                    return None
                
                col_idx = self._get_column_idx(current_column)
                self.logger.info(f"Current column of block {block} is {current_column}")
            
            # Move to a column near the selected column
            if col_idx == 0:
                col_idx = col_idx + 1
            elif col_idx == len(columns) - 1:
                col_idx = col_idx - 1
            else:
                col_idx = col_idx + random.choice([-1, 1])
            
            new_column = columns[col_idx]
            self.logger.info(f"Moving block {block} to column {new_column} instead of {action.actual_parameters[1].object().name}")
            new_action = self._get_specific_action(block, new_column)
            return new_action
    
    # Check preconditions of an action and apply effects if the action is applicable
    def is_action_legal(self, plan_action: unified_planning.plans.plan.ActionInstance, state=None):
        if state is None:
            state = self.state
        action = plan_action.action
        grounded_args = {param.name: str(grounded_param) for param, grounded_param in zip(action.parameters, plan_action.actual_parameters)}
        preconditions = action.preconditions
        for node in preconditions:
            if not self._check_value(node, grounded_args, state=state):
                self.logger.debug(f"Precondition not met: {node} with args {grounded_args}")
                return False
        
        return True
    
    # Action needs to come from a plan generated by a unified_planning planner
    def apply_action(self, plan_action: unified_planning.plans.plan.ActionInstance, can_fail=True):
        if not self.is_action_legal(plan_action):
            self.logger.info(f"Action {plan_action} is not legal.")
            return False, "not legal"
        
        action = plan_action.action
        # Grounded args is a dictionary mapping parameter names to actual parameter values taken from the plan action
        # e.g. {'b1': 'r'}, where b1 is the generic parameter from the action and 'r' is the value it takes in the current plan action
        grounded_args = {param.name: str(grounded_param) for param, grounded_param in zip(action.parameters, plan_action.actual_parameters)}
        self.logger.debug(f"Grounded args {grounded_args}")
        
        if can_fail and random.random() < self.fail_probability:
            # Simulate a failure
            self.logger.info(f"Simulating action failure")
            new_action = self.fail_action(plan_action)
            if new_action is not None:
                self.logger.info(f"Applying action {new_action}")
                return self.apply_action(new_action, can_fail=False)[0], str(new_action) # Do not allow further failures
            else:
                return True, "no-op"
        
        # Update render state
        block_id = grounded_args['b1'].capitalize() 
        block = self.render_state.get_block(block_id)
        column = grounded_args['c1']
        top_block = self._top_block(column)
        if top_block is None:
            # Block is moved to an empty column
            column_idx = self._get_column_idx(column)
            self.logger.debug(f"Renderer: Moving block {block} ({block_id}) to column {column} at index {column_idx}")
            self.render_state.action_move_column(block, column_idx)
        else:
            # Block is moved on top of another block
            top_block_id = top_block.capitalize()
            top_block = self.render_state.get_block(top_block_id)
            self.logger.debug(f"Renderer: Moving block {block} ({block_id}) on top of block {top_block} ({top_block_id})")
            self.render_state.action_move_specific(block, top_block)
        
        effects = action.effects
        
        state_before_action = copy.deepcopy(self.state)
        for effect in effects:
            tmp_state = copy.deepcopy(state_before_action)
            self.logger.debug("Applying effect:", effect)
            self._apply_effect(effect, grounded_args, tmp_state)
            self._update_state_dict(state_before_action, tmp_state)
        
        return True, "success"
