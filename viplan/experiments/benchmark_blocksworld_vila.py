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

from viplan.planning.blocksworld_simulator import BlocksworldSimulator
from viplan.code_helpers import get_logger, load_vlm, get_unique_id

block_templates = {
    'r': 'red block',
    'g': 'green block',
    'b': 'blue block',
    'y': 'yellow block',
    'o': 'orange block',
    'p': 'purple block',
}

def parse_json_output(output):
    json_start = output.find('{')
    json_end = output.rfind('}')
    vlm_plan = json.loads(output[json_start:json_end + 1])
    return vlm_plan

def get_plan_action(env, json_plan):
    try:
        first_action = json_plan['plan'][0]
        return env._get_specific_action(block=first_action['parameters']['block'], column=first_action['parameters']['column'], action_name=first_action['action'])
    except KeyError as e:
        print(f"KeyError: {e}")
        print(f"JSON plan: {json_plan}")
        return None

# TODO move this to code_helpers and replace in benchmark_blocksworld_plan.py too
def load_problem(domain_file, problem_file, logger, root_path, fail_probability, seed=1, use_gpu_rendering=True):
    
    reader = PDDLReader()
    problem = reader.parse_problem(domain_file, problem_file)
    env = BlocksworldSimulator(problem, root_path=root_path, logger=logger, seed=seed, use_gpu_rendering=use_gpu_rendering, fail_probability=fail_probability)
    
    return env, problem

def get_goal_str(env):
    goal_fluents = env.goal_fluents
    blocks = list(env.problem.objects(env.problem.user_type('block')))
    goal_string = ""

    for block in blocks:
        block_fluents = [f for f in goal_fluents if f.args[0].object() == block]        
        goal_string += f"The {block_templates[block.name]}"
        
        for i, block_fluent in enumerate(block_fluents):
            name = block_fluent.fluent().name
            if name == 'incolumn':
                goal_string += f" needs to be in the column with label '{block_fluent.args[1].object().name}'"
            elif name == 'on':
                goal_string += f" needs to be on top of the {block_templates[block_fluent.args[1].object().name]}"
            elif name == 'clear':
                goal_string += f" needs to be the topmost block in its column"
            else:
                raise Exception(f"Unknown fluent {name} in goal")
            goal_string += "," if i < len(block_fluents) - 2 else " and" if i == len(block_fluents) - 2 else ". "
            
        goal_string += "\n"
        
    return goal_string

def planning_loop(env, model, base_prompt, logger, max_steps=50):
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
        logger.info(f"Environment state before action:\n{env}")
        prompt = base_prompt.replace("{previous_actions}", json.dumps(previous_actions))
        logger.info(f"Prompt:\n{prompt}")
        
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
            action = get_plan_action(env, vlm_plan)
            problem_results['actions'].append({'action': str(action)})
            logger.info(f"First action: {action}")

            success, info = env.apply_action(action)
            problem_results['actions'][-1]['success'] = success
            problem_results['actions'][-1]['info'] = info
            wrong_parameters = False
        except Exception as e:
            logger.error(f"Could not apply action: {e}")
            problem_results['actions'].append({'action': str(action) if action is not None else 'unknown action', 'success': False, 'info': info})
            if type(e) == AssertionError:
                wrong_parameters = True
            success = False
        
        # Failsafe for actions that don't exist
        try:
            available_actions = [ a.name for a in env.problem.actions ]
            if vlm_plan['plan'][0]['action'] not in available_actions or action is None:
                logger.warn(f"Action {vlm_plan['plan'][0]['action']} does not exist in the environment.")
                previous_actions.append({'action': vlm_plan['plan'][0]['action'], 'outcome': 'action does not exist'})
            else:
                if wrong_parameters is not None and wrong_parameters:
                    previous_actions.append({'action': str(action.action.name), 'parameters': vlm_plan['plan'][0]['parameters'], 'outcome': 'parameters incorrectly specified', 'info': info})
                else:
                    previous_actions.append({'action': str(action.action.name), 'parameters': {str(p): str(v) for p, v in zip([p.type for p in action.action.parameters], action.actual_parameters)}, 'outcome': 'success' if success else 'failure', 'info': info})
        except Exception as e:
            logger.error(f"Something went wrong (e.g. plan is empty): {e}")
            break
        
        logger.info(f"Action outcome: {'success' if success else 'failure'}")
        logger.info(f"Previous actions: {previous_actions}")
        logger.info(f"Environment state after action:\n{env}")
        problem_results['previous_actions'] = previous_actions

        max_steps -= 1

    if env.goal_reached:
        logger.info("Goal reached!")
        problem_results['completed'] = True
    
    return problem_results
        

def main(
    problems_dir: os.PathLike,
    domain_file: os.PathLike,
    root_path: os.PathLike,
    model_name: str,
    prompt_path: os.PathLike,
    seed: int = 1,
    output_dir: os.PathLike = None,
    hf_cache_dir: os.PathLike = None,
    log_level =' info',
    use_gpu_rendering: bool = True,
    fail_probability: float = 0.0,
    max_steps: int = 25,
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
    problem_files = [os.path.join(problems_dir, f) for f in os.listdir(problems_dir) if f.endswith(".pddl")]

    for problem_file in problem_files:
        logger.info(f"Loading problem {problem_file}")
        try:
            env, problem = load_problem(domain_file, problem_file, logger, root_path, seed=seed, use_gpu_rendering=use_gpu_rendering, fail_probability=fail_probability)
        except Exception as e:
            logger.error(f"Could not load problem {problem_file}: {e}")
            results[problem_file] = None
            continue
        
        goal_string = get_goal_str(env)
        logger.info(f"Goal: {goal_string}")
        problem_prompt = base_prompt.replace("{goal_string}", goal_string)
        
        # Run planning loop
        logger.info("Starting planning loop...")
        problem_results = planning_loop(env, model, problem_prompt, logger, max_steps=max_steps)
        results[problem_file] = problem_results
        
    
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
        'fail_probability': fail_probability,
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