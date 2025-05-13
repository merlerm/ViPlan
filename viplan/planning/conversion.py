import os
import itertools
import numpy as np

block_id = {
    'R': 1,
    'G': 2,
    'B': 3,
    'Y': 4,
    'P': 5,
    'O': 6,
}

block_letter = {
    1: 'R',
    2: 'G',
    3: 'B',
    4: 'Y',
    5: 'P',
    6: 'O',
}

# Conversion functions
def _sort_column(blocks, on):
    # Create a mapping from the lower block to the block above it.
    below_to_top = {v: k for k, v in on.items()}
    
    # Find the bottom block: a block that is not on top of another block.
    # (Assuming there's exactly one such block per column.)
    bottom_candidates = [b for b in blocks if b not in on]
    if not bottom_candidates:
        # In case there is an error or cycle, just return the original list.
        return blocks
    bottom = bottom_candidates[0]
    
    # Build the sorted list starting from the bottom.
    sorted_stack = [bottom]
    while bottom in below_to_top:
        bottom = below_to_top[bottom]
        sorted_stack.append(bottom)
    return sorted_stack

def _sort_columns(state):
    graph = {}
    indegree = {}
    
    # Process predicates.
    # For leftof(c1, c2): add edge c1 -> c2.
    # For rightof(c1, c2): add edge c2 -> c1.
    for predicate in ['leftof', 'rightof']:
        for args, value in state.get(predicate, {}).items():
            # Only process if the predicate is true.
            if not value:
                continue
            c1, c2 = args.split(',')
            # Set up the edge direction based on the predicate.
            if predicate == 'leftof':
                a, b = c1, c2  # a should come before b
            else:  # rightof
                a, b = c2, c1  # b must be to the right => a comes before b

            if a not in graph:
                graph[a] = []
            if b not in graph:
                graph[b] = []
                
            graph[a].append(b)
            indegree[b] = indegree.get(b, 0) + 1
            if a not in indegree:
                indegree[a] = 0

    # Ensure all columns mentioned anywhere are represented.
    all_columns = set()
    for predicate in ['leftof', 'rightof']:
        for args in state.get(predicate, {}):
            c1, c2 = args.split(',')
            all_columns.add(c1)
            all_columns.add(c2)
    for col in all_columns:
        if col not in graph:
            graph[col] = []
        if col not in indegree:
            indegree[col] = 0

    # Topologically sort the graph.
    # Start with all nodes with no incoming edges.
    available = sorted([node for node, deg in indegree.items() if deg == 0])
    ordered_columns = []

    while available:
        # Choose the next available column (alphabetically first if tie).
        node = available.pop(0)
        ordered_columns.append(node)
        
        # For every column that must come after the current node, reduce its indegree.
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                available.append(neighbor)
        available.sort()  # maintain alphabetical order if multiple choices exist.

    # In case a cycle exists or some columns were not reached via constraints,
    # append any missing columns sorted alphabetically.
    missing = all_columns - set(ordered_columns)
    if missing:
        ordered_columns.extend(sorted(missing))
    
    return ordered_columns

def predicates_to_numpy(state, use_letters=False):
    # Check which blocks are in which columns
    blocks_in_columns = {}
    inColumn = state['incolumn']
    for args, value in inColumn.items():
        if value:
            block, column = args.split(',')
            blocks_in_columns[column] = [block] if column not in blocks_in_columns else blocks_in_columns[column] + [block]
    # Check which blocks are on top of which other blocks
    on = state['on']
    blocks_on = {}
    for args, value in on.items():
        if value:
            block_on_top, block_on_bottom = args.split(',')
            blocks_on[block_on_top] = block_on_bottom
    # Sort the blocks in each column
    sorted_columns = {column: _sort_column(blocks, blocks_on) for column, blocks in blocks_in_columns.items()}
    
    # Sort the columns
    column_order = _sort_columns(state)
    
    # Determine dimensions based on actual data.
    n_cols = len(column_order)
    n_blocks = 0
    for col in column_order:
        if col in sorted_columns:
            n_blocks = max(n_blocks, len(sorted_columns[col]))
            
    # Create numpy state matrix of shape (n_cols, n_blocks)
    np_state = np.zeros((n_cols, n_blocks), dtype=int)
    for i, col in enumerate(column_order):
        if col not in sorted_columns:
            continue
        column_blocks = sorted_columns[col]
        for j, block in enumerate(column_blocks):
            np_state[i, j] = block_id[block.upper()]
            
    if use_letters:
        np_state = np_state.astype(str)
        for i in range(n_cols):
            for j in range(n_blocks):
                np_state[i, j] = block_letter[int(np_state[i, j])] if np_state[i, j] != '0' else '-'
    return np_state

# Utilities for converting between BIRD file strings and states
def bird_to_state(bird_string):
    columns = bird_string.split(".")[0].split("-")
    state = np.zeros((7, 6), dtype=int)
    for i, column in enumerate(columns):
        for j, block in enumerate(column):
            state[i, j] = block_id[block] if block != '0' else 0
            
    return state

# Check all unordered alternatives for columns to find an alternative image
def state_to_bird(state):
    bird_string = ""
    for i in range(7):
        for j in range(6):
            bird_string += block_letter[state[i, j]] if state[i, j] != 0 else "0"
        bird_string += "-"
    return bird_string[:-1] + ".jpg"

# Find an alternative image for a bird string without considering the order of columns
def find_bird_alternative(bird_string, data_path):
    if os.path.exists(os.path.join(data_path, bird_string)):
        return bird_string
    
    bird_columns = bird_string.split(".")[0].split("-")
    permutations = list(itertools.permutations(bird_columns))
    for perm in permutations:
        perm_string = "-".join(perm) + ".jpg"
        if os.path.exists(os.path.join(data_path, perm_string)):
            return perm_string
    return None