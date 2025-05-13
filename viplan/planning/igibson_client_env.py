import io
import re
import json
import base64
import requests
from unified_planning.model import Problem

from PIL import Image

from viplan.code_helpers import get_logger

class iGibsonClient():
    
    def __init__(self, 
                 task: str,
                 scene_id: str,
                 base_url: str,
                 problem: Problem,
                 instance_id: int = 0,
                 logger = get_logger('info'),
    ):
        
        self.task = task
        self.scene_id = scene_id
        self.instance_id = instance_id
        self.problem = problem
        self.base_url = base_url
        self.logger = logger
        self.all_objects = {str(self.problem.user_types[type_]): list(self.problem.objects(self.problem.user_types[type_])) for type_ in range(len(self.problem.user_types))}
        
        self.reset()
        self.state, self.img = self._get_state_and_img()
        
    def __str__(self):
        return str(self.state)
    
    def reset(self):
        # Update all_objects in case problem has changed
        self.all_objects = {str(self.problem.user_types[type_]): list(self.problem.objects(self.problem.user_types[type_])) for type_ in range(len(self.problem.user_types))}

        payload = {
            "task": self.task,
            "scene_id": self.scene_id,
            "instance_id": self.instance_id
        }
        
        response = requests.post(f"{self.base_url}/reset", json=payload)  
        
        if response.status_code == 200:
            data = response.json()
            self.logger.info(f"\nReset successful: {data['success']}")
            
            # Update the state and image after reset
            self.state, self.img = self._get_state_and_img()
            if self.state is None or self.img is None:
                self.logger.error("Failed to get initial state or image after reset")
                return False
            
            return True
        else:
            self.logger.error(f"Reset failed with status code {response.status_code}")
            self.logger.error(response.text)
            return False
        
    def _get_state_and_img(self):
        response = requests.get(f"{self.base_url}/get_state")
        
        if response.status_code == 200:
            data = response.json()
            state = data['symbolic_state']

            # Enforce here that all fluents exist as keys, even if just as empty dictionaries
            state = self._add_missing_keys(state)
            
            img_data = base64.b64decode(data['image'])
            img = Image.open(io.BytesIO(img_data))
            
            return state, img
        else:
            self.logger.error(f"Get state failed with status code {response.status_code}")
            self.logger.error(response.text)
            return None, None

    def _add_missing_keys(self, state):
        for fluent in self.problem.initial_values:
            name = fluent.fluent().name 
            if name not in state.keys():
                state[name] = {}
        return state
        
    @property
    def priviledged_predicates(self):
        
        if self.state is not None:
            inside_preds = self.state.get('inside', {})
            inside_preds = {k: v for k, v in inside_preds.items() if v}
            
            reachable_preds = self.state.get('reachable', {})
            reachable_preds = {k: v for k, v in reachable_preds.items() if v}
            
            inside_keys = [key.split(',')[0] for key in inside_preds.keys()]
            inside_containers = [key.split(',')[1] for key in inside_preds.keys()]
            reachable_keys = list(reachable_preds.keys())
            inside_but_not_reachable = {'inside': set([(inside_keys[i], inside_containers[i]) for i in range(len(inside_keys)) if inside_keys[i] not in reachable_keys])}
            return inside_but_not_reachable
        else:
            self.logger.error("State is None, cannot get priviledged predicates")
            return None
    
    def _get_visible_objects(self):
        response = requests.get(f"{self.base_url}/get_visible_objects")
        
        if response.status_code == 200:
            data = response.json()
            visible_objects = data['objects']
            return visible_objects
        else:
            self.logger.error(f"Get visible objects failed with status code {response.status_code}")
            self.logger.error(response.text)
            return None
        
    @property
    def visible_predicates(self):
        visible_objects = self._get_visible_objects()
        if visible_objects is not None:
            visible_preds = {}
            for predicate in self.state:
                for args in self.state[predicate].keys():
                    if all([arg in visible_objects for arg in args.split(',')]):
                        if predicate not in visible_preds:
                            visible_preds[predicate] = {}
                        visible_preds[predicate][args] = self.state[predicate][args]
            return visible_preds
        else:
            self.logger.error("Failed to get visible objects, cannot get visible predicates")
            return None
        
    @property
    def goal_fluents(self):
        goal_predicates = self.problem.goals[0] # Assuming there are just a set of "ands" in the goal
        assert goal_predicates.is_and() or goal_predicates.is_fluent_exp(), "Goal structure was not a simple conjunction of fluents"
        
        if goal_predicates.is_and():
            for goal in goal_predicates.args:
                if goal.is_not():
                    goal = goal.args[0]
                assert goal.is_fluent_exp(), "Goal structure was not a simple conjunction of fluents"
            goal_fluents = goal_predicates.args
        else:
            goal_fluents = [goal_predicates]

        print("goal_fluents: ", goal_fluents)
        return goal_fluents
            
    @property
    def goal_reached(self):
        goal_fluents = self.goal_fluents
        self.logger.debug(f"Goal fluents: {goal_fluents}")
        self.logger.debug(f"State: {self.state}")
        if self.state is None:
            self.logger.error("State is None, cannot check goal reached")
            return False
            
        # Enforce here that all fluents exist as keys, even if just as empty dictionaries
        # should avoid issues in line `sat = self.state[fluent_name][",".join(fluent_args)] == value`
        self.state = self._add_missing_keys(self.state)
        
        for fluent in goal_fluents:
            value = True
            if fluent.is_not():
                fluent = fluent.args[0]
                value = False
            fluent_name = fluent.fluent().name
            fluent_args = [a.object().name for a in fluent.args]
            
            # Safeguard for missing keys in the state
            state_value = self.state.get(fluent_name, {}).get(",".join(fluent_args), None)
            if state_value is None:
                self.logger.warning(f"Goal fluent {str(fluent)} not found in state")
                return False
            sat = state_value == value
            if not sat:
                self.logger.debug(f"Goal fluent {str(fluent)} not satisfied")
                return False
        
        self.logger.info("Goal reached")
        return True
    
    # Recursively check the truth value of a node, which can contain either a fluent (predicate with arguments) or a logical operator
    # The state argument is needed because when applying effects the truth value of a fluent needs to be checked from the state BEFORE the effect was applied, since self.state will have incomplete updates
    def _check_value(self, node, grounded_args, state=None):
        if state is None:
            state = self._add_missing_keys(self.state) # add_missing_keys added for extra safety
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
            if not ','.join(args) in state[fluent_name]:
                # This is for example on(cabinet1, cabinet1) which should be false but is not in the state
                return False
            value = state[fluent_name][','.join(args)]
            return value if isinstance(value, bool) else value.is_true()
        else:
            raise ValueError("Unknown node type", node)
        
    def render(self):
        # Keep the render function for compatibility with the other envs (blocksworld)
        return self.img
    
    def apply_action(self, action: str, params: list):
        
        payload = {
            "action": action,
            "params": params
        }
        
        response = requests.post(f"{self.base_url}/execute_action", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            self.logger.info(f"\nAction legal: {data['success']}")
            self.logger.info(f"\nAction execution successful: {data['info']}")
            
            # Update the state and image after action execution
            # self.state, self.img = self._get_state_and_img() # shouldn't be needed
            state = data['symbolic_state']
            self.state = self._add_missing_keys(state) 
            img_data = base64.b64decode(data['image'])
            self.img = Image.open(io.BytesIO(img_data))
            
            if self.state is None or self.img is None:
                self.logger.error("Failed to get state or image after action execution")
                return False, data['info']
            return data['success'], data['info']
        if response.status_code == 400:
            error_detail = response.json().get("detail", "Invalid request")
            self.logger.warning(f"Action rejected by server: {error_detail}")
            return False, error_detail
        else:
            self.logger.error(f"Action execution failed with status code {response.status_code}")
            self.logger.error(response.text)
            return False, f'server returned {response.status_code}'