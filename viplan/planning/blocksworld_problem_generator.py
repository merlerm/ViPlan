import os
import sys
import fire
import json
import random
import tempfile
import networkx as nx

from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader
from viplan.code_helpers import get_logger

def sample_blocks(blocks_config):
    return random.sample(blocks_config['colors'], k=blocks_config['num'])

def generate_state_graph(sampled_blocks, col_config, max_init_col_height, logger):

    # Initialize the graph with a node for each sampled block and a node for each column
    G = nx.DiGraph()
    G.add_nodes_from(sampled_blocks, type='block')
    G.add_nodes_from(col_config['names'], type='column')
    
    # For each block, add an edge either to the column or to another block
    # Make sure that the column has enough space -> each column can have at most max_init_col_height blocks
    
    for block in sampled_blocks:
        # Sample a column with enough space
        col = random.choice(col_config['names'])
        while G.in_degree(col) >= max_init_col_height:
            col = random.choice(col_config['names'])
        G.add_edge(block, col)
        logger.debug(f"Block {block} added to column {col}")
        
        # If there are other blocks in the column, we need to add an edge to the top block (the one with 0 in-degree)
        blocks_in_col = [n for n in G.predecessors(col)]
        if len(blocks_in_col) == 1:
            continue # Only the block itself is in the column
        if blocks_in_col:
            top_block = blocks_in_col[-2] # The last block is the one we just added
            assert G.in_degree(top_block) == 0, f"Top block {top_block} has in-degree {G.in_degree(top_block)}"
            if top_block != block:
                G.add_edge(block, top_block)
                logger.debug(f"Block {block} added on top of block {top_block}")
    
    return G

def graph_to_predicate_list(G, col_config):
    predicates = {
        'on': [],
        'clear': [],
        'inColumn': [],
        'rightOf': [],
        'leftOf': [],
    }
    
    for node in G.nodes:
        if G.nodes[node]['type'] == 'block':
            if G.in_degree(node) == 0:
                predicates['clear'].append(f'{node}')
            else:
                predecessors = list(G.predecessors(node))
                predicates['on'].append((predecessors[0], node))

            for col in col_config['names']:
                if G.has_edge(node, col):
                    predicates['inColumn'].append((node, col))
                    break
                
    # Column order
    for i in range(len(col_config['names']) - 1):
        predicates['leftOf'].append((col_config['names'][i], col_config['names'][i+1]))
        predicates['rightOf'].append((col_config['names'][i+1], col_config['names'][i]))
        
    return predicates

def get_problem_string(problem_config, sampled_blocks, init_predicates, goal_predicates):
    
    problem_str = f"""

(define (problem {problem_config['problem_name']})
  (:domain blocksworld)
  
  (:objects 
    {" ".join(sampled_blocks)} - block
    {" ".join(problem_config['col_config']['names'])} - column
  )
  
  (:init
"""

    for predicate in init_predicates:
        for args in init_predicates[predicate]:
            problem_str += f"\n    ({predicate} {' '.join(args)})"
        problem_str += "\n"
        
    problem_str += "  )"
    
    problem_str += "\n  (:goal"
    problem_str += "\n    (and"
    for predicate in goal_predicates:
        for args in goal_predicates[predicate]:
            problem_str += f"\n      ({predicate} {' '.join(args)})"
        problem_str += "\n"
        
    problem_str += "    )"
    problem_str += "\n  )"
    problem_str += "\n)"
    
    return problem_str.strip()

def is_valid_problem(domain_file, problem_file, min_plan_length=1, max_plan_length=100):
    reader = PDDLReader()
    problem = reader.parse_problem(domain_file, problem_file)
    
    with OneshotPlanner(problem_kind=problem.kind) as planner:
        result = planner.solve(problem)
        if not result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
            return False, None
        
    if len(result.plan.actions) < min_plan_length or len(result.plan.actions) > max_plan_length:
        return False, result.plan
    
    return True, result.plan


def generate_problem(
    problem_config_file,
    problem_output_file,
    domain_file,
    logger=get_logger('info'),
    max_attempts=100,
    idx=None):
        
    with open(problem_config_file, 'r') as f:
        problem_config = json.load(f)
        
        if idx is not None:
            problem_config['problem_name'] = f"{problem_config['problem_name']}_{idx}"
    
    sampled_blocks = sample_blocks(problem_config['blocks_config'])
    logger.info(f"Sampled blocks: {sampled_blocks}")
    
    G = generate_state_graph(sampled_blocks, problem_config['col_config'], problem_config['max_init_col_height'], logger)
    init_predicates = graph_to_predicate_list(G, problem_config['col_config'])
    logger.info(f"Generated initial state with predicates: {init_predicates}")
    
    valid = False
    while not valid and max_attempts > 0:
        max_attempts -= 1
        goal_G = generate_state_graph(sampled_blocks, problem_config['col_config'], len(sampled_blocks), logger) # All blocks in the same column 
        while G == goal_G:
            goal_G = generate_state_graph(sampled_blocks, problem_config['col_config'], len(sampled_blocks), logger)
        goal_predicates = graph_to_predicate_list(goal_G, problem_config['col_config'])
        # Discard rightOf and leftOf predicates as they can't change
        goal_predicates.pop('rightOf')
        goal_predicates.pop('leftOf')
        
        problem_str = get_problem_string(problem_config, sampled_blocks, init_predicates, goal_predicates)
        assert problem_str[0] == '(', "Problem string does not start with a parenthesis"

        with tempfile.NamedTemporaryFile(mode='w+', delete=True) as f:
            f.write(problem_str)
            f.seek(0)
            problem_file = f.name
            
            min_plan_length = problem_config.get('min_plan_length', 1)
            max_plan_length = problem_config.get('max_plan_length', 100)
            
            valid, plan = is_valid_problem(domain_file, problem_file, min_plan_length=min_plan_length, max_plan_length=max_plan_length)
            if not valid:
                reason = "Goal not reachable" if plan is None else f"Plan length {len(plan.actions)} not in [{min_plan_length}, {max_plan_length}]"
                logger.warning(f"Generated invalid goal: {reason}. {max_attempts} attempts left.")
    if valid:
        logger.info(f"Generated goal state with predicates: {goal_predicates}")
        logger.info(f"Goal is reachable with a plan of length {len(plan.actions)}")
        with open(problem_output_file, 'w') as f:
            f.write(problem_str)
        logger.info(f"Problem generated successfully and saved to {problem_output_file}")
        
        metadata = {
            'blocks': sampled_blocks,
            'init_predicates': init_predicates,
            'goal_predicates': goal_predicates,
            'reference_plan': list(map(str, plan.actions)),
        }
        
        return metadata
        
    else:
        logger.error(f"Could not generate a valid problem after {max_attempts} attempts.")
        sys.exit(1)
        
        
def generate_problem_batch(problem_config_file, domain_file, output_dir, num_problems, log_level='info'):
    logger = get_logger(log_level)
    
    with open(problem_config_file, 'r') as f:
        problem_config = json.load(f)
        
    metadata = {}
    
    problems_done = 0
    while problems_done < num_problems:
        logger.info(f"Generating problem {problems_done+1}/{num_problems}")
        problem_output_file = os.path.join(output_dir, f"{problem_config['problem_name']}_{problems_done}.pddl")
        problem_metadata = generate_problem(problem_config_file, problem_output_file, domain_file, logger=logger, max_attempts=100, idx=problems_done)
        metadata[f"{problem_config['problem_name']}_{problems_done}"] = problem_metadata
        
        problems = os.listdir(output_dir)
        for problem in problems:
            # Check if any of the other problems are equivalent to the one we just generated
            if problem == f"{problem_config['problem_name']}_{problems_done}.pddl":
                continue
            with open(os.path.join(output_dir, problem), 'r') as f:
                other_problem = f.read()
            with open(problem_output_file, 'r') as f:
                new_problem = f.read()
            if other_problem == new_problem:
                logger.warning(f"Generated problem is equivalent to {problem}. Regenerating.")
                break
        else:
            problems_done += 1
        
    logger.info(f"Generated {num_problems} problems and saved them to {output_dir}")
    
    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)
        logger.info("Metadata saved to metadata.json")

if __name__ == '__main__':
    fire.Fire(generate_problem_batch)
