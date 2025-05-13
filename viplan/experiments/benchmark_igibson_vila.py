import os
import sys
import fire
import json
import copy

import torch
import random
import transformers

from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader

from PIL import Image
import matplotlib.pyplot as plt

from viplan.planning.igibson_client_env import iGibsonClient
from viplan.code_helpers import get_logger, load_vlm, get_unique_id

preds_templates = {
    'reachable': "the {0} is reachable by the agent",
    'holding':   "the agent is holding the {0}",
    'open':      "the {0} is open",
    'ontop':     "the {0} is on top of the {1}",
    'inside':    "the {0} is inside the {1}",
    'nextto':    "the {0} is next to the {1}",
}

goal_templates = {
    'reachable': {
        True:  "the {0} needs to be reachable by the agent",
        False: "the {0} needs to be unreachable by the agent"
    },
    'holding': {
        True:  "the agent needs to be holding the {0}",
        False: "the agent must not be holding the {0}"
    },
    'open': {
        True:  "the {0} needs to be open",
        False: "the {0} needs to be closed"
    },
    'ontop': {
        True:  "the {0} needs to be on top of the {1}",
        False: "the {0} must not be on top of the {1}"
    },
    'inside': {
        True:  "the {0} needs to be inside the {1}",
        False: "the {0} must not be inside the {1}"
    },
    'nextto': {
        True:  "the {0} needs to be next to the {1}",
        False: "the {0} must not be next to the {1}"
    }
}

def parse_json_output(output):
    json_start = output.find('{')
    json_end = output.rfind('}')
    vlm_plan = json.loads(output[json_start:json_end + 1])
    return vlm_plan

"""
def get_goal_str(env):
    goal_fluents = env.goal_fluents
    goal_string = ""

    for fluent in goal_fluents:
        name = fluent.fluent().name
        args = fluent.args

        descr = goal_templates[str(name)].format(*[arg.object().name for arg in args])
        goal_string += descr + "\n"
    
    goal_string = goal_string.strip()
    return goal_string
"""

def get_goal_str(env):
    goal_fluents = env.goal_fluents
    goal_string = ""

    for fluent in goal_fluents:
        value = True
        if fluent.is_not():
            value = False
            fluent = fluent.args[0]
        name = fluent.fluent().name
        args = fluent.args

        descr = goal_templates[str(name)][value].format(*[arg.object().name for arg in args])
        goal_string += descr + "\n"
    
    goal_string = goal_string.strip()
    return goal_string
    
def get_priviledged_predicates_str(predicates):
    priviledged_string = ""
    for pred, values in predicates.items():
        for args in values:
            if args:
                name = pred
                args = args
                descr = preds_templates[str(name)].format(*args)
                priviledged_string += descr + "\n"
    priviledged_string = priviledged_string.strip()
    return priviledged_string

def planning_loop(env, model, base_prompt, problem, logger, max_steps=50):
    previous_actions = []
    problem_results = {
        'plans': [],
        'actions': [],
        'previous_actions': previous_actions,
        'completed': False,
    }
    
    initial_max_steps = copy.deepcopy(max_steps)

    while not env.goal_reached and max_steps > 0:
        logger.info(f"Step {initial_max_steps - max_steps + 1}")
        logger.info(f"Environment state before action:\n{env.state}")
        prompt = base_prompt.replace("{previous_actions}", json.dumps(previous_actions))
        priviledged_preds = env.priviledged_predicates
        
        if priviledged_preds is not None:
            if all(priviledged_preds[predicate] == {} for predicate in priviledged_preds):
                prompt = prompt.replace("## Additional information","")
                prompt = prompt.replace("{priviledged_info}","")
            else:
                prompt = prompt.replace("{priviledged_info}", get_priviledged_predicates_str(priviledged_preds))
        else:
            prompt = prompt.replace("## Additional information","")
            prompt = prompt.replace("{priviledged_info}","")
            
        logger.debug(f"Prompt:\n{prompt}")
        
        outputs = model.generate(prompts=[prompt], images=[env.render()], return_probs=False)
        logger.info("VLM output: " + outputs[0])
        try:
            vlm_plan = parse_json_output(outputs[0])
        except json.JSONDecodeError as e:
            logger.error(f"Could not parse VLM output: {e}")
            logger.error(f"VLM output: {outputs[0]}")
            break
        problem_results['plans'].append(vlm_plan)
        if 'explanation' in vlm_plan:
            logger.info(f"VLM CoT: {vlm_plan['explanation']}")

        
        
        try:
            action = None
            info = None
            first_action = vlm_plan['plan'][0]
            action_params = first_action['parameters']
            action = f"{first_action['action']}({', '.join([str(p) for p in action_params])})"
            problem_results['actions'].append({'action': action})
            logger.info(f"First action: {action}")
            success, info = env.apply_action(action=first_action['action'], params=action_params)
        except Exception as e:
            logger.error(f"Unexpected error applying action: {e}")
            problem_results['actions'].append({'action': action if action is not None else 'unknown action', 'success': False})
            success = False
            info = None
            wrong_parameters = False  # can't assume it's input-related
        else:
            problem_results['actions'][-1]['success'] = success
            problem_results['actions'][-1]['info'] = info
            wrong_parameters = not success
        
        # Failsafe for actions that don't exist
        try:
            available_actions = [ a.name for a in problem.actions ]
            logger.debug(f"Available actions: {available_actions}, first action: {first_action['action']}")
            if first_action['action'] not in available_actions or action is None:
                logger.warn(f"Action {first_action['action']} does not exist in the environment.")
                previous_actions.append({'action': first_action['action'], 'outcome': 'action does not exist'})
            else:
                if wrong_parameters is not None and wrong_parameters:
                    previous_actions.append({'action': first_action['action'], 'parameters': vlm_plan['plan'][0]['parameters'], 'outcome': 'parameters incorrectly specified'})
                else:
                    previous_actions.append({'action': first_action['action'], 'parameters': action_params, 'outcome': 'executed' if success else 'failed'})
        except Exception as e:
            logger.error(f"Something went wrong (e.g. plan is empty): {e}")
            break
        
        logger.info(f"Action outcome: {'executed' if success else 'failed'}")
        if info is not None:
            logger.info(f"Info about action outcome: {info}")
        logger.info(f"Previous actions: {previous_actions}")
        logger.info(f"Environment state after action:\n{env.state}")
        problem_results['previous_actions'] = previous_actions

        max_steps -= 1

    if env.goal_reached: 
        logger.info("Goal reached!")
        problem_results['completed'] = True
    
    return problem_results
        

def main(
    problems_dir: os.PathLike,
    domain_file: os.PathLike,
    base_url: str,
    model_name: str,
    prompt_path: os.PathLike,
    seed: int = 1,
    output_dir: os.PathLike = None,
    hf_cache_dir: os.PathLike = None,
    log_level ='info',
    max_steps: int = 10,
    **kwargs):
    
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    
    logger = get_logger(log_level=log_level)
    unique_id = get_unique_id(logger)
        
    if hf_cache_dir is None:
        hf_cache_dir = os.environ.get("HF_HOME", None)
        logger.debug(f"Using HF cache dir: {hf_cache_dir}")
    
    model = load_vlm(model_name, hf_cache_dir=hf_cache_dir, logger=logger, **kwargs)
    
    unified_planning.shortcuts.get_environment().credits_stream = None # Disable planner printouts
    
    with open(prompt_path, 'r') as f:
        base_prompt = f.read()
    
    results = {}
    metadata = os.path.join(problems_dir, "metadata.json")
    assert os.path.exists(metadata), f"Metadata file {metadata} not found"
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    problem_files = [problem for problem in metadata.keys()]
    problem_files = [f"{problems_dir}/{problem}" for problem in problem_files]
    
    for problem_file in problem_files:
        logger.info(f"Loading problem {problem_file}")
        
        reader = PDDLReader()
        problem = reader.parse_problem(domain_file, problem_file)
        task = metadata[os.path.basename(problem_file)]['activity_name']
        scene_instance_pairs = metadata[os.path.basename(problem_file)]['scene_instance_pairs']
        for scene_id, instance_id in scene_instance_pairs:
            env = iGibsonClient(task=task, scene_id=scene_id, instance_id=instance_id, problem=problem, base_url=base_url, logger=logger)
            env.reset() # Reset = send a request to the server to re-initialize the task (also needed when switching tasks)
            
            goal_string = get_goal_str(env)
            logger.info(f"Goal: {goal_string}")
            problem_prompt = base_prompt.replace("{goal_string}", goal_string)
            
            # Run planning loop
            logger.info("Starting planning loop...")
            problem_results = planning_loop(env, model, problem_prompt, problem, logger, max_steps=max_steps)
            results[f"{problem_file}_{scene_id}_{instance_id}"] = problem_results
    
    # Compute some statistics
    total_actions = 0
    total_success = 0
    total_failed = 0
    total_tasks_completed = 0
    
    for problem_file, problem_results in results.items():
        if problem_results is None:
            continue
        total_actions += len(problem_results['actions'])
        total_success += sum([1 for a in problem_results['actions'] if a.get('success', False)])
        total_failed += sum([1 for a in problem_results['actions'] if not a.get('success', False)])
        total_tasks_completed += 1 if problem_results['completed'] else 0
        
    action_success_rate = total_success / total_actions if total_actions > 0 else 0
    action_failure_rate = total_failed / total_actions if total_actions > 0 else 0
    task_completion_rate = total_tasks_completed / len(results) if len(results) > 0 else 0
    
    results['statistics'] = {
        'total_actions': total_actions,
        'total_success': total_success,
        'total_failed': total_failed,
        'total_tasks_completed': total_tasks_completed,
        'action_success_rate': action_success_rate,
        'action_failure_rate': action_failure_rate,
        'task_completion_rate': task_completion_rate,
    }
    
    results['metadata'] = {
        'model': model_name,
        'seed': seed,
        'prompt_path': prompt_path,
        'max_steps': max_steps,
        'job_id': unique_id,
    }

    logger.info(f"Action success rate: {action_success_rate}")
    logger.info(f"Action failure rate: {action_failure_rate}")
    logger.info(f"Task completion rate: {task_completion_rate}")
    
    if output_dir is None:
        output_dir = os.curdir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{unique_id}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f)
    logger.info(f"Results saved to {output_file}")
    
if __name__ == "__main__":
    fire.Fire(main)