import os
import copy
import fire
import json
import torch
import random
import tempfile
import transformers

from typing import List

from collections import deque

from unified_planning.shortcuts import *
from unified_planning.io import PDDLReader
from unified_planning.environment import get_environment

from viplan.code_helpers import get_logger, parse_output, get_unique_id
from viplan.planning.blocksworld_simulator import BlocksworldSimulator

# Standard templates

predicate_questions = {
    'on': 'Is the {b1} on top of the {b2}?',
    'clear': 'Is the {b} the topmost of its column?',
    'incolumn': 'Is the {b} in the {c}?',
    'rightof': 'Is the {c1} to the right of the {c2}?',
    'leftof': 'Is the {c1} to the left of the {c2}?',
}

block_templates = {
    'r': 'red block',
    'g': 'green block',
    'b': 'blue block',
    'y': 'yellow block',
    'o': 'orange block',
    'p': 'purple block',
}

column_templates = {
    'c1': 'column labeled "c1"',
    'c2': 'column labeled "c2"',
    'c3': 'column labeled "c3"',
    'c4': 'column labeled "c4"',
    'c5': 'column labeled "c5"',
} 

column_templates_text = {
    'c1': 'first column',
    'c2': 'second column',
    'c3': 'third column',
    'c4': 'fourth column',
    'c5': 'fifth column',
}

# Single predicate questions (from a given set)

def get_questions(predicates, block_templates, column_templates):
    questions = {}
    
    for pred in predicates:
        key = list(pred.keys())[0]
        predicate = key.split(" ")[0]
        value = pred[key]
        args = key.split(" ")[1].split(",")
        
        if predicate == 'on':
            b1 = block_templates[args[0]]
            b2 = block_templates[args[1]]
            questions[predicate + " " + ",".join(args)] = (predicate_questions[predicate].format(b1=b1, b2=b2), value)
        elif predicate == 'clear':
            b = block_templates[args[0]]
            questions[predicate + " " + ",".join(args)] = (predicate_questions[predicate].format(b=b), value)
        elif predicate == 'incolumn':
            b = block_templates[args[0]]
            c = column_templates[args[1]]
            questions[predicate + " " + ",".join(args)] = (predicate_questions[predicate].format(b=b, c=c), value)
        elif predicate == 'rightof':
            c1 = column_templates[args[0]]
            c2 = column_templates[args[1]]
            questions[predicate + " " + ",".join(args)] = (predicate_questions[predicate].format(c1=c1, c2=c2), value)
        elif predicate == 'leftof':
            c1 = column_templates[args[0]]
            c2 = column_templates[args[1]]
            questions[predicate + " " + ",".join(args)] = (predicate_questions[predicate].format(c1=c1, c2=c2), value)
        else:
            raise ValueError(f"Unknown predicate {predicate}")
    
    return questions

def get_predicates_for_question(node, grounded_args, top_level=True, default_value=True):
    result = []

    if not node.is_fluent_exp():
        
        if node.is_not():
            child = node.args[0]
            exps = get_predicates_for_question(child, grounded_args, False)
            if len(exps) > 1:
                raise NotImplementedError("Not implemented for multiple expressions")
            for key, value in exps[0].items():
                result.append({key: not value})
        else:
            for child in node.args:
                exps = get_predicates_for_question(child, grounded_args, False)
                result.extend(exps)

    elif node.is_fluent_exp():
        fluent_name = node.fluent().name
        arg_names = [str(arg) for arg in node.args]
        actual_args = [str(grounded_args[arg]) for arg in arg_names]
        args_key = ",".join(actual_args)
        key = fluent_name + " " + args_key
        value = default_value # by default assume precondition is asking for predicate to be default_value (for effects it can also be False)
        bool_value = value if isinstance(value, bool) else value.is_true()
        result = {key: bool_value}

    else:
        raise ValueError("Unknown node type", node)

    return [result] if type(result) is dict else result

# Predicate sets for preconditions and effects

def get_preconditions_predicates(preconditions, grounded_args):
    preconditions_predicates = []
    for precondition in preconditions:
        preds = get_predicates_for_question(precondition, grounded_args)
        preconditions_predicates.extend(preds)
    return preds

def get_effect_predicates(env, effect, grounded_args, previous_state):
    """
    Applies effect to the state.
    For non-conditional effects, it uses get_predicates_for_question to extract predicates from the fluent.
    For conditional effects, it checks the condition using env.check_value on previous_state.
    Returns a list of predicates gathered from the effect applications.
    """
    all_preds = []

    if effect.is_forall():
        assert len(effect.forall) == 1, "Only single forall supported"
        var = effect.forall[0]
        for value in env.all_objects[str(var.type)]:
            grounded_args[var.name] = str(value)
            if effect.is_conditional():
                condition = effect.condition
                # For conditional effects, use the previous state.
                if not env._check_value(condition, grounded_args, previous_state):
                    continue
            preds = get_predicates_for_question(effect.fluent, grounded_args, top_level=False, default_value=effect.value)
            if isinstance(preds, dict):
                all_preds.append(preds)
            else:
                all_preds.extend(preds)

    elif effect.is_conditional():
        condition = effect.condition
        # For conditional effects, check condition against previous_state.
        if env._check_value(condition, grounded_args, previous_state):
            preds = get_predicates_for_question(effect.fluent, grounded_args, top_level=False, default_value=effect.value)
            if isinstance(preds, dict):
                all_preds.append(preds)
            else:
                all_preds.extend(preds)

    else:
        preds = get_predicates_for_question(effect.fluent, grounded_args, top_level=False, default_value=effect.value)
        if isinstance(preds, dict):
            all_preds.append(preds)
        else:
            all_preds.extend(preds)

    return all_preds

def get_effects_predicates(env, effects, grounded_args, previous_state):
    all_preds = []
    for effect in effects:
        preds = get_effect_predicates(env, effect, grounded_args, previous_state)
        all_preds.extend(preds)
    return all_preds

# Create new problem with updated state to replan

def update_problem(state, problem):
    
    def get_new_problem_fluent(new_problem, fluent):
        for new_fluent in new_problem.initial_values:
            if str(new_fluent) == str(fluent):
                return new_fluent
        return None
       
    new_problem = copy.deepcopy(problem)
    
    up_env = get_environment()
    expr_manager = up_env.expression_manager
    
    for fluent in problem.initial_values:
        
        name = fluent.fluent().name
        args = fluent.args
        args_str = ",".join([str(arg) for arg in args])
        value = problem.initial_values[fluent].is_true()
        state_value = state[name][args_str]
        if value != state_value:
            assert problem.initial_values[fluent].is_bool_constant()
            new_fluent = get_new_problem_fluent(new_problem, fluent)
            if new_fluent is None:
                raise ValueError(f"Fluent {fluent} not found in new_problem")
            
            new_problem.initial_values[new_fluent] = expr_manager.true_expression if state_value else expr_manager.false_expression
            # print(f"Updating {new_fluent} from {problem.initial_values[fluent]} to {new_problem.initial_values[new_fluent]}")

    return new_problem

# Query model with set of questions (either for predicates or effects)

def cast_to_yes_no(parsed_answer, logger): 
    if parsed_answer is None: # empty answer or invalid format
        return "invalid answer"
    
    if not parsed_answer.startswith("yes") and not parsed_answer.startswith("no"):
        for word in parsed_answer.split(" "):
            if word.startswith("yes"):
                parsed_answer = "yes"
                logger.debug(f"Found 'yes' in answer: {parsed_answer}")
                break
            elif word.startswith("no"):
                parsed_answer = "no"
                logger.debug(f"Found 'no' in answer: {parsed_answer}")
                break
    elif parsed_answer.startswith("yes"):
        parsed_answer = "yes"
    elif parsed_answer.startswith("no"):
        parsed_answer = "no"
        
    return parsed_answer

def ask_vlm(questions, image, model, base_prompt, logger, env, **kwargs):
    base_prompt = open(base_prompt, 'r').read()
    prompts = [base_prompt + q[0] for q in questions.values()]
    images = [image for _ in questions]

    outputs = model.generate(prompts=prompts, images=images, return_probs=True, **kwargs)
    
    results = {}
    for j, (key) in enumerate(questions.keys()):
        answer, yes_prob, no_prob = outputs[j]
        original_output = copy.deepcopy(answer)
        answer, tags_found = parse_output(answer, answer_tags=["answer", "explanation"])
        parsed_answer = answer['answer'] if tags_found and 'answer' in answer else answer
        parsed_answer = parsed_answer.strip().lower().rstrip('.,!?') if isinstance(parsed_answer, str) else parsed_answer
        parsed_answer = cast_to_yes_no(parsed_answer, logger)
        
        parsed_explanation = answer['explanation'] if tags_found and 'explanation' in answer else None

        logger.info(f"Q: {questions[key][0]}, A: {parsed_answer}, Yes: {yes_prob:.2f}, No: {no_prob:.2f}")
        if parsed_explanation is not None:
            logger.info(f"Explanation (CoT): {parsed_explanation}")
        
        # Answer match = the answer is what the PDDL model would expect -> if false, then replan
        answer_match = parsed_answer == 'yes' if questions[key][1] else parsed_answer == 'no'     
        
        # State match = the answer is what is actually true in the environment
        # Can differ from PDDL if the action failed in the environment and the model correctly notices
        # e.g. block is dropped in the wrong column, PDDL expects it to be in the correct column but the model sees it dropped
        # -> state match is used to compute action accuracy, as it is the metric checking the model's actual performance
        
        predicate, args = key.split(" ")
        state_value = env.state[predicate][args]
        if (state_value and parsed_answer == 'yes') or (not state_value and parsed_answer == 'no'):
            state_match = True
        else:
            state_match = False
        logger.info(f"Actual predicate value: {predicate} {args} = {state_value}, model answer: {parsed_answer}, state match: {state_match}")
        logger.info(f"PDDL expected value: {predicate} {args} = {questions[key][1]}, model answer: {parsed_answer}, answer match: {answer_match}")
        
        results[key] = (parsed_answer, yes_prob, no_prob, parsed_explanation, answer_match, original_output, state_match)
    
    # All correct = all predicates are correct according to the PDDL model -> no replan needed
    # State match can only be used for metrics -> in the real world, the ground truth is not known
    results['all_correct'] = all([results[result][4] for result in results])
    results['all_state_correct'] = all([results[k][6] for k in results if k not in ['all_correct', 'all_state_correct']])
                    
    return results

# TODO delete this and adapt ask_vlm to work with the LLM
def ask_llm(questions, state_str, model, base_prompt, logger, **kwargs):
    base_prompt = open(base_prompt, 'r').read()
    if '{image}' in base_prompt:
        logger.warning("Base prompt contains image tag, removing as we are not using images")
        base_prompt = base_prompt.replace("{image}", "")
    
    base_prompt += "The current state is:\n" + state_str + "\n"
        
    prompts = [base_prompt + q[0] for q in questions.values()]

    outputs = model.get_completion(prompts=prompts, **kwargs)
    
    results = {}
    
    for j, (key) in enumerate(questions.keys()):
        answer, yes_prob, no_prob = outputs[j], 0, 0 # TODO add probabilities for LLM
        original_output = copy.deepcopy(answer)
        answer, tags_found = parse_output(answer, answer_tags=["answer", "explanation"])
        parsed_answer = answer['answer'] if tags_found and 'answer' in answer else answer
        parsed_answer = parsed_answer.strip().lower() if isinstance(parsed_answer, str) else parsed_answer
        parsed_answer = cast_to_yes_no(parsed_answer, logger)
        
        parsed_explanation = answer['explanation'] if tags_found and 'explanation' in answer else None

        logger.info(f"Q: {questions[key][0]}, A: {parsed_answer}, Yes: {yes_prob:.2f}, No: {no_prob:.2f}")
        
        answer_match = parsed_answer == 'yes' if questions[key][1] else parsed_answer == 'no'                
        results[key] = (parsed_answer, yes_prob, no_prob, parsed_explanation, answer_match, original_output)

    results['all_correct'] = all([results[result][4] for result in results])
                    
    return results

# Dynamic planning helpers

def update_vlm_state(vlm_state, results):
    changed = []
    for key, result in results.items():
        if key == 'all_correct' or key == 'all_state_correct':
            continue
        pred, args = key.split(" ")
        assert result[0] == 'yes' or result[0] == 'no', f"VLM gave unexpected answer {result[0]}"
        new_value = True if result[0] == 'yes' else False
        if vlm_state[pred][args] != new_value:
            vlm_state[pred][args] = new_value
            changed_str = f"{pred} {args} to {new_value}" # Str for better logging, later we can use the key
            changed.append(changed_str)

    return vlm_state, changed

def check_preconditions(env, preconditions, grounded_args, model, base_prompt, logger, text_only=False):
    precondition_preds = get_preconditions_predicates(preconditions, grounded_args)
    questions = get_questions(precondition_preds, block_templates, column_templates_text if text_only else column_templates)
    if text_only:
        results = ask_llm(questions, str(env), model, base_prompt, logger)
    else:
        results = ask_vlm(questions, env.render(), model, base_prompt, logger, env)
    
    return results

def check_effects(env, effects, grounded_args, model, base_prompt, previous_state, logger, text_only=False):
    effect_preds = get_effects_predicates(env, effects, grounded_args, previous_state)
    questions = get_questions(effect_preds, block_templates, column_templates_text if text_only else column_templates)
    if text_only:
        results = ask_llm(questions, str(env), model, base_prompt, logger)
    else:
        results = ask_vlm(questions, env.render(), model, base_prompt, logger, env)
    
    return results

def check_action(env, action, vlm_state, model, base_prompt, logger, text_only=False):
    preconditions = action.action.preconditions
    effects = action.action.effects
    grounded_params = {param.name: str(value) for param, value in zip(action.action.parameters, action.actual_parameters)}
    previous_state = copy.deepcopy(env.state)
    logger.info("Environment state before action\n" + str(env))
    
    preconditions_results = check_preconditions(env, preconditions, grounded_params, model, base_prompt, logger, text_only=text_only)
    vlm_state, changed = update_vlm_state(vlm_state, preconditions_results)
    if len(changed) > 0:
        logger.debug("VLM state changed after preconditions:", changed)

    # If the preconditions are not satisfied according to the PDDL model, the action can not be taken
    if not preconditions_results['all_correct']:
        logger.warning("Preconditions not satisfied")
        return False, preconditions_results, None, False, None
    
    legal, info = env.apply_action(action)
    # VLM thought the action was legal, but it was not
    if not legal:
        logger.warning("Action was not legal")
        return False, preconditions_results, None, False, info
    
    logger.info("Environment state after action\n" + str(env))
    
    effects_results = check_effects(env, effects, grounded_params, model, base_prompt, previous_state, logger, text_only=text_only)
    vlm_state, changed = update_vlm_state(vlm_state, effects_results)
    if len(changed) > 0:
        logger.debug("VLM state changed after effects:", changed)
    
    precond_all_correct = preconditions_results['all_correct']
    effects_all_correct = effects_results['all_correct']
    all_correct = precond_all_correct and effects_all_correct
    
    # Effects state correct can be different from all_correct if there was a failure in the environment and the VLM detected it
    precond_state_correct = preconditions_results['all_state_correct']
    effects_state_correct = effects_results['all_state_correct']
    all_state_correct = precond_state_correct and effects_state_correct
    
    return all_correct, preconditions_results, effects_results, all_state_correct, info

# State enumeration

# TODO why is this needed? It's the same as get_questions, they should be merged
def get_enumeration_questions(predicates, block_templates, column_templates):
    questions = {}
    for pred in predicates:
        predicate = str(pred)
        for args_str in predicates[pred]:
            args = tuple(args_str.split(','))
            value = predicates[pred][args_str]
            
            if predicate == 'on':
                b1 = block_templates[args[0]]
                b2 = block_templates[args[1]]
                questions[predicate + " " + args_str] = (predicate_questions[predicate].format(b1=b1, b2=b2), value)
            elif predicate == 'clear':
                b = block_templates[args[0]]
                questions[predicate + " " + args_str] = (predicate_questions[predicate].format(b=b), value)
            elif predicate == 'incolumn':
                b = block_templates[args[0]]
                c = column_templates[args[1]]
                questions[predicate + " " + args_str] = (predicate_questions[predicate].format(b=b, c=c), value)
            elif predicate == 'rightof':
                c1 = column_templates[args[0]]
                c2 = column_templates[args[1]]
                questions[predicate + " " + args_str] = (predicate_questions[predicate].format(c1=c1, c2=c2), value)
            elif predicate == 'leftof':
                c1 = column_templates[args[0]]
                c2 = column_templates[args[1]]
                questions[predicate + " " + args_str] = (predicate_questions[predicate].format(c1=c1, c2=c2), value)
            else:
                raise ValueError(f"Unknown predicate {predicate}")
    
    return questions

def get_enumeration_results(env, model, questions, base_prompt, logger, batch_size=64):
    responses = []
    base_prompt = open(base_prompt, 'r').read()
    q_list = [ base_prompt + "\n" + question[0] for question in questions.values() ]
    
    for i in range(0, len(questions), batch_size):
        logger.info(f"Processing questions {i} to {min(i + batch_size, len(q_list))} of {len(q_list)}")
        batch_prompts = q_list[i:i + min(batch_size, len(q_list) - i)]
        img = env.render()
        batch_images = [img] * len(batch_prompts)
        batch_responses = model.generate(prompts=batch_prompts, images=batch_images, return_probs=True)
        responses.extend(batch_responses)
        
    assert len(responses) == len(q_list), "Some answers were not generated."
    
    # Convert to a format that works with update_vlm_state
    results = {}
    for i, (question, response) in enumerate(zip(questions, responses)):
        logger.debug(f"Question: {question}, Response: {response}")
        question_str = questions[question][0]
        answer = 'yes' if questions[question][1] else 'no'
        response, tags_found = parse_output(response[0], answer_tags=["answer", "explanation"])
        parsed_answer = response['answer'] if tags_found and 'answer' in response else response
        parsed_answer = parsed_answer.strip().lower().rstrip('.,!?') if isinstance(parsed_answer, str) else parsed_answer
        parsed_answer = cast_to_yes_no(parsed_answer, logger)
        logger.debug(f"Parsed answer: {parsed_answer}")
        
        results[question] = (parsed_answer, answer.lower())
    
    return results

def compute_enumeration_metrics(results):
    enum_results = {
        'accuracy': 0,
        'yes_accuracy': 0,
        'yes_correct': 0,
        'yes_total': 0,
        'no_accuracy': 0,
        'no_correct': 0,
        'no_total': 0,
        'predicates': {}
    }
    # For enumeration, there is no expected value from the PDDL model (as we're testing everything), so the accuracy is already based on the environment
    for question, (answer, expected_answer) in results.items():
        if answer == expected_answer:
            enum_results['accuracy'] += 1
            if expected_answer == 'yes':
                enum_results['yes_correct'] += 1
            else:
                enum_results['no_correct'] += 1
                
        if expected_answer == 'yes':
            enum_results['yes_total'] += 1
        else:
            enum_results['no_total'] += 1
        
        predicate = question.split(" ")[0]
        if predicate not in enum_results['predicates']:
            enum_results['predicates'][predicate] = {
                'accuracy': 0,
                'yes_accuracy': 0,
                'yes_correct': 0,
                'yes_total': 0,
                'no_accuracy': 0,
                'no_correct': 0,
                'no_total': 0
            }
        if answer == expected_answer:
            enum_results['predicates'][predicate]['accuracy'] += 1
            if expected_answer == 'yes':
                enum_results['predicates'][predicate]['yes_correct'] += 1
            else:
                enum_results['predicates'][predicate]['no_correct'] += 1
                
        if expected_answer == 'yes':
            enum_results['predicates'][predicate]['yes_total'] += 1
        else:
            enum_results['predicates'][predicate]['no_total'] += 1
    
    enum_results['accuracy'] /= len(results)
    enum_results['yes_accuracy'] = (enum_results['yes_correct'] / enum_results['yes_total']) if enum_results['yes_total'] > 0 else None
    enum_results['no_accuracy'] = (enum_results['no_correct'] / enum_results['no_total']) if enum_results['no_total'] > 0 else None
    
    for predicate in enum_results['predicates']:
        enum_results['predicates'][predicate]['accuracy'] /= (enum_results['predicates'][predicate]['yes_total'] + enum_results['predicates'][predicate]['no_total'])
        enum_results['predicates'][predicate]['yes_accuracy'] = (enum_results['predicates'][predicate]['yes_correct'] / enum_results['predicates'][predicate]['yes_total']) if enum_results['predicates'][predicate]['yes_total'] > 0 else None
        enum_results['predicates'][predicate]['no_accuracy'] = (enum_results['predicates'][predicate]['no_correct'] / enum_results['predicates'][predicate]['no_total']) if enum_results['predicates'][predicate]['no_total'] > 0 else None
        
    return enum_results

# Dynamic planning
def get_plan(problem, logger):
    result = None
    try:
        with OneshotPlanner(problem_kind=problem.kind) as planner:
            # This is needed to avoid temporary file conflicts created in cwd from the planner in job arrays
            with tempfile.TemporaryDirectory() as td:
                old_cwd = os.getcwd()
                try:
                    os.chdir(td)  # Change to temporary directory
                    result = planner.solve(problem)
                finally:
                    os.chdir(old_cwd)  # Restore original directory
            
            if result.status == up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
                logger.debug("Fast Downward returned: %s" % result.plan)
            else:
                logger.warning("No plan found.")
    except Exception as e:
        logger.warning(f"Planner crashed with error: {e}")

    return result

def check_plan(env, 
               plan,
               problem,
               vlm_state, # Initial state as perceived by the VLM
               model, 
               base_prompt, 
               logger,
               replan=False,
               text_only=False,
               max_actions=20,
               enumerate_replan=False,
               enum_batch_size=64):
    
    all_correct = True
    results = []
    replans = []
    action_queue = deque(plan.actions)
    most_recent_action = None
    while action_queue and len(results) < max_actions:
        action = action_queue.popleft()
        
        logger.info(f"Applying action {action}")
        
        try:
            action_correct, preconditions_results, effects_results, action_state_correct, action_info = check_action(env, action, vlm_state, model, base_prompt, logger, text_only=text_only)
        except Exception as e:
            logger.warning(f"Error while checking action {action}: {e}")
            action_correct = False
            preconditions_results = {}
            effects_results = None
            action_state_correct = False
            action_info = None
            break
            
        results.append({
            'action': str(action),
            'action_correct': action_correct,
            'action_state_correct': action_state_correct,
            'preconditions_results': preconditions_results,
            'effects_results': effects_results,
            'action_info': action_info,
        })
        if not action_correct:
            if not preconditions_results['all_correct']:
                reason = "Preconditions not satisfied"
                failed_results = preconditions_results
            elif effects_results is None:
                reason = "Action was not legal"
                failed_results = {}
            elif not effects_results['all_correct']:
                reason = "Not all effects were observed as expected"
                failed_results = effects_results
            else:
                reason = "Unknown"
            logger.warning(f"Action {action} failed: {reason}")
            if reason == "Action was not legal" and str(action) == str(most_recent_action):
                logger.warning("Action was not legal, but it was the same as the most recent action. Stopping as we're likely in a loop.")
                break
            try:
                all_correct = False

                if replan:
                    replans.append({})
                    logger.info("Replanning from newly observed state")
                    
                    if enumerate_replan:
                        # Enumerate the predicates in the new state
                        questions = get_enumeration_questions(env.state, block_templates, column_templates)
                        enum_results = get_enumeration_results(env, model, questions, base_prompt, logger, batch_size=enum_batch_size)
                        enum_metrics = compute_enumeration_metrics(enum_results)
                        replans[-1]['enum_results'] = enum_results
                        replans[-1]['enum_metrics'] = enum_metrics
                        
                        vlm_state, changed = update_vlm_state(copy.deepcopy(env.state), enum_results)
                        # No need to update vlm state if not enumerating, as the effect results are already updated in check_action
                else:
                    break
            except Exception as e:
                logger.warning(f"Error while updating VLM state: {e}")
                all_correct = False
                break
                
            new_problem = update_problem(vlm_state, env.problem)
            plan_result = get_plan(new_problem, logger)
            if plan_result is None:
                logger.warning("Breaking out of episode due to error in the planner")
                break # Exit episode
            elif plan_result.status != up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
                logger.warning("No plan found after replanning")
                break
            else:
                logger.info("Replan found")
                new_plan = plan_result.plan
                action_queue = deque(new_plan.actions)
                replans[-1].update({
                    'step': len(results),
                    'actions': [str(a) for a in new_plan.actions]
                })                    
            
        if len(action_queue) == 0:
            logger.info("All actions completed")
            break
        if len(results) >= max_actions:
            logger.warning("Max actions reached")
            break
        
        most_recent_action = copy.deepcopy(action)
        
    goal_reached = env.goal_reached
    logger.info(f"Goal reached: {goal_reached}")
    
    if all_correct and not goal_reached:
        logger.warning(f"All actions executed correctly, but goal not reached")

    return all_correct, results, replans, action_queue, goal_reached

def compute_metrics(results, logger):
    # task accuracy = task was fully completed, 
    # action_accuracy = fraction of individual actions that were correctly predicted (in full)
    # predicate_accuracy = fraction of predicates that were correctly predicted
    # macro_predicate_accuracy = fraction of predicates that were correctly predicted, equally weighted independently of the number of predicates
    # fail_ratio = fraction of problems that never had a plan (wrong initial state)

    task_accuracy = sum([results[problem]['goal_reached'] for problem in results if 'goal_reached' in results[problem]]) / len(results)
    problem_stats = {}
    predicate_stats = {}
    
    # Problem stats for action accuracy
    for problem in results:
        problem_stats[problem] = {}
        try:
            # Action accuracy is computed on the actual state correctness (e.g. did the VLM correctly answer as to what it was seeing)
            problem_stats[problem]['action_correct'] = sum([action['action_state_correct'] for action in results[problem]['action_results'] if 'action_state_correct' in action])
            problem_stats[problem]['action_total'] = len([action for action in results[problem]['action_results'] if 'action_state_correct' in action])
            # print(f"Problem: {problem}, actions: {n_actions}, action_accuracy: {action_accuracy}")
            problem_stats[problem]['action_total'] += len(results[problem]['remaining_actions'])
            problem_stats[problem]['remaining_actions'] = results[problem]['remaining_actions']
            problem_stats[problem]['action_accuracy'] = problem_stats[problem]['action_correct'] / problem_stats[problem]['action_total'] if problem_stats[problem]['action_total'] > 0 else 0
            problem_stats[problem]['failed'] = False
            # print(f"Problem: {problem}, actions: {n_actions}, action_accuracy: {action_accuracy} (after remaining actions {len(results[problem]['remaining_actions'])})")
        except Exception as e:
            problem_stats[problem]['action_correct'] = 0
            problem_stats[problem]['action_total'] = 1 # count this as a first action that failed to normalize the metric
            problem_stats[problem]['failed'] = True
            print(f"Problem {problem} had no actions, likely never started due to wrong initial state")
            continue
    logger.debug(f"Problem stats: {problem_stats}")
    
    # Predicate stats for predicate accuracy
    for problem in results:
        # First add the enumeration results
        if 'initial_state_enum' in results[problem]:
            logger.debug(f"Adding initial state enumeration results for problem {problem}")
            for predicate in results[problem]['initial_state_enum']['statistics']['predicates']:
                if predicate not in predicate_stats:
                    predicate_stats[predicate] = {
                        'accuracy': 0,
                        'yes_accuracy': 0,
                        'yes_correct': 0,
                        'yes_total': 0,
                        'no_accuracy': 0,
                        'no_correct': 0,
                        'no_total': 0
                    }
                predicate_stats[predicate]['yes_correct'] += results[problem]['initial_state_enum']['statistics']['predicates'][predicate]['yes_correct']
                predicate_stats[predicate]['yes_total'] += results[problem]['initial_state_enum']['statistics']['predicates'][predicate]['yes_total']
                predicate_stats[predicate]['no_correct'] += results[problem]['initial_state_enum']['statistics']['predicates'][predicate]['no_correct']
                predicate_stats[predicate]['no_total'] += results[problem]['initial_state_enum']['statistics']['predicates'][predicate]['no_total']
        else:
            logger.info(f"No initial state enumeration results for problem {problem}")
        
        # Check if there is enumeration in the replans
        if 'replans' in results[problem] and results[problem]['replans'] is not None:
            logger.debug(f"Adding replan enumeration results for problem {problem}")
            for replan in results[problem]['replans']:
                if 'enum_metrics' in replan:
                    for predicate in replan['enum_metrics']['predicates']:
                        if predicate not in predicate_stats:
                            predicate_stats[predicate] = {
                                'accuracy': 0,
                                'yes_accuracy': 0,
                                'yes_correct': 0,
                                'yes_total': 0,
                                'no_accuracy': 0,
                                'no_correct': 0,
                                'no_total': 0
                            }
                        predicate_stats[predicate]['yes_correct'] += replan['enum_metrics']['predicates'][predicate]['yes_correct']
                        predicate_stats[predicate]['yes_total'] += replan['enum_metrics']['predicates'][predicate]['yes_total']
                        predicate_stats[predicate]['no_correct'] += replan['enum_metrics']['predicates'][predicate]['no_correct']
                        predicate_stats[predicate]['no_total'] += replan['enum_metrics']['predicates'][predicate]['no_total']
        else:
            logger.info(f"No replan enumeration results for problem {problem}")

            def update_predicate_stats(predicate_stats, results):
                for key, res in results.items():
                    if key in ('all_correct', 'all_state_correct'):
                        continue
                    predicate = key.split(" ")[0]
                    model_answer = res[0]
                    state_correct = res[6]
                    if predicate not in predicate_stats:
                        predicate_stats[predicate] = {
                            'accuracy': 0,
                            'yes_accuracy': 0,
                            'yes_correct': 0,
                            'yes_total': 0,
                            'no_accuracy': 0,
                            'no_correct': 0,
                            'no_total': 0
                        }
                    if model_answer == 'yes':
                        if state_correct:
                            predicate_stats[predicate]['yes_correct'] += 1
                        predicate_stats[predicate]['yes_total'] += 1
                    elif model_answer == 'no':
                        if not state_correct:
                            predicate_stats[predicate]['no_correct'] += 1
                        predicate_stats[predicate]['no_total'] += 1
                    else:
                        logger.warning(f"Unexpected answer {model_answer} for predicate {predicate}")
                        continue

                for action in results[problem]['action_results']:
                    update_predicate_stats(predicate_stats, action.get('preconditions_results', {}))
                    update_predicate_stats(predicate_stats, action.get('effects_results', {}))
        
    action_accuracy = sum([problem_stats[problem]['action_correct'] for problem in problem_stats]) / sum([problem_stats[problem]['action_total'] for problem in problem_stats])
    fail_ratio = sum([problem_stats[problem]['failed'] for problem in problem_stats]) / len(problem_stats)
            
    for predicate in predicate_stats:
        predicate_stats[predicate]['correct'] = predicate_stats[predicate]['yes_correct'] + predicate_stats[predicate]['no_correct']
        predicate_stats[predicate]['total'] = predicate_stats[predicate]['yes_total'] + predicate_stats[predicate]['no_total']
        predicate_stats[predicate]['accuracy'] = predicate_stats[predicate]['correct'] / predicate_stats[predicate]['total'] if predicate_stats[predicate]['total'] > 0 else 0
        
        predicate_stats[predicate]['yes_accuracy'] = predicate_stats[predicate]['yes_correct'] / predicate_stats[predicate]['yes_total'] if predicate_stats[predicate]['yes_total'] > 0 else 0
        predicate_stats[predicate]['no_accuracy'] = predicate_stats[predicate]['no_correct'] / predicate_stats[predicate]['no_total'] if predicate_stats[predicate]['no_total'] > 0 else 0
    
    logger.debug(f"Predicate stats: {predicate_stats}")
    predicate_accuracy = sum([predicate_stats[predicate]['correct'] for predicate in predicate_stats]) / sum([predicate_stats[predicate]['total'] for predicate in predicate_stats])
    macro_predicate_accuracy = sum([predicate_stats[predicate]['accuracy'] for predicate in predicate_stats]) / len(predicate_stats) if len(predicate_stats) > 0 else 0

    return predicate_accuracy, macro_predicate_accuracy, action_accuracy, task_accuracy, problem_stats, predicate_stats, fail_ratio    

def load_problem(domain_file, problem_file, logger, root_path, fail_probability, seed=1, use_gpu_rendering=True):
    
    reader = PDDLReader()
    problem = reader.parse_problem(domain_file, problem_file)
    env = BlocksworldSimulator(problem, root_path=root_path, logger=logger, seed=seed, use_gpu_rendering=use_gpu_rendering, fail_probability=fail_probability)
    
    return env, problem

def main(
    problems_dir: os.PathLike,
    domain_file: os.PathLike,
    root_path: os.PathLike,
    model_name: str,
    prompt_path: os.PathLike,
    seed: int = 1,
    output_dir: os.PathLike = None,
    hf_cache_dir: os.PathLike = None,
    log_level ='info',
    use_gpu_rendering: bool = True,
    replan: bool = True, # Try to replan if an action fails
    text_only: bool = False, # Instead of asking based on the image, ask based on a textual description
    fail_probability: float = 0.0, # Probability of action failure
    enumerate_initial_state: bool = True, # Enumerate initial state predicates (instead of using oracle)
    enumerate_replan: bool = True, # Enumerate predicates before replanning if there is a failure
    enum_batch_size: int = 64, # Batch size for enumeration
    **kwargs):
    
    # Ensure deterministic behavior (in theory)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    random.seed(seed)
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    # torch.use_deterministic_algorithms(True)
    
    logger = get_logger(log_level=log_level)
    unique_id = get_unique_id(logger)
        
    if hf_cache_dir is None:
        hf_cache_dir = os.environ.get("HF_HOME", None)
        logger.debug(f"Using HF cache dir: {hf_cache_dir}")
        
    unified_planning.shortcuts.get_environment().credits_stream = None # Disable planner printouts

    # Load model
    logger.info(f"Using GPU: {torch.cuda.get_device_name()}." if torch.cuda.is_available() else "Using CPU.")
    if torch.cuda.is_available() and ('A100' in torch.cuda.get_device_name() or 'H100' in torch.cuda.get_device_name() or 'H200' in torch.cuda.get_device_name()):
        use_flash_attn = True
    else:
        use_flash_attn = False
    logger.info(f"Use flash attention: {use_flash_attn}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    if text_only:
        logger.info("Using LLM for text-only evaluation")
        from viplan.models import HuggingFaceLLM
        model = HuggingFaceLLM(model_name, cache_dir=hf_cache_dir, logger=logger, temperature=0, device=device, dtype=dtype, use_flash_attention=use_flash_attn, **kwargs)
    else:
        from viplan.code_helpers import load_vlm
        model = load_vlm(model_name, cache_dir=hf_cache_dir, logger=logger, temperature=0, device=device, dtype=dtype, use_flash_attention=use_flash_attn, **kwargs)
    logger.info(f"Loaded model {model_name} on device {device} with dtype {dtype}")
    
    # Find problem files
    problem_files = [os.path.join(problems_dir, f) for f in os.listdir(problems_dir) if f.endswith(".pddl")]
    
    results = {}
    for problem_file in problem_files:
        results[problem_file] = {}
        logger.info(f"Loading problem {problem_file}")
        try:
            env, problem = load_problem(domain_file, problem_file, logger, root_path, seed=seed, use_gpu_rendering=use_gpu_rendering, fail_probability=fail_probability)
        except Exception as e:
            logger.error(f"Could not load problem {problem_file}: {e}")
            results[problem_file] = {
                'all_correct': False,
                'action_results': None,
                'replans': None,
                'remaining_actions': None
            }
            continue
        
        if enumerate_initial_state:
            try:
                _ = get_plan(problem, logger) # For some reason we need to plan with the problem before updating it or it breaks ???
                
                logger.info(f"Original state:\n{str(env)}")
                logger.info("Enumerating initial state predicates")
                predicates = env.state
                questions = get_enumeration_questions(predicates, block_templates, column_templates)
                enum_results = get_enumeration_results(env, model, questions, prompt_path, logger, batch_size=enum_batch_size)
                stats = compute_enumeration_metrics(enum_results)
                results[problem_file]['initial_state_enum'] = {'results': enum_results, 'statistics': stats}
                logger.info(f"Enumeration accuracy: {stats['accuracy']:.2f}, Yes accuracy: {stats['yes_accuracy']:.2f}, No accuracy: {stats['no_accuracy']:.2f}")
                
                vlm_state, changed = update_vlm_state(copy.deepcopy(env.state), enum_results)
                logger.info(f"VLM state changed: {changed}")
                problem = update_problem(vlm_state, problem)
                
            except Exception as e:
                logger.error(f"Could not enumerate initial state: {e}")
                import traceback
                logger.error(traceback.format_exc())
                results[problem_file].update({
                'all_correct': False,
                'action_results': None,
                'replans': None,
                'remaining_actions': None
            })
                continue
        else:
            vlm_state = copy.deepcopy(env.state) # Oracle state
        
        plan_result = get_plan(problem, logger)
        if plan_result is None:
            logger.warning("Breaking out of episode due to error in the planner")
            results[problem_file].update({
                'all_correct': False,
                'action_results': None,
                'replans': None,
                'remaining_actions': None
            })
            continue
        elif plan_result.status != up.engines.PlanGenerationResultStatus.SOLVED_SATISFICING:
            logger.warning("No plan found.")
            results[problem_file].update({
                'all_correct': False,
                'action_results': None,
                'replans': None,
                'remaining_actions': None
            })
            continue
        plan = plan_result.plan
        
        all_correct, action_results, replans, action_queue, goal_reached = check_plan(env, plan, problem, vlm_state, model, prompt_path, logger, replan=replan, text_only=text_only, enumerate_replan=enumerate_replan, enum_batch_size=enum_batch_size)
        results[problem_file].update({
            'all_correct': all_correct,
            'goal_reached': goal_reached,
            'action_results': action_results,
            'replans': replans,
            'remaining_actions': [str(a) for a in action_queue]
        })
    
    predicate_accuracy, macro_predicate_accuracy, action_accuracy, task_accuracy, problem_stats, predicate_stats, fail_ratio = compute_metrics(results, logger)
    logger.info(f"Predicate accuracy: {predicate_accuracy:.2f}, Macro predicate accuracy: {macro_predicate_accuracy:.2f}, Action accuracy: {action_accuracy:.2f}, Task accuracy: {task_accuracy:.2f}")
    logger.info(f"Fail ratio: {fail_ratio:.2f}")
    results['problem_stats'] = problem_stats
    results['predicate_stats'] = predicate_stats
    results['predicate_accuracy'] = predicate_accuracy
    results['macro_predicate_accuracy'] = macro_predicate_accuracy
    results['action_accuracy'] = action_accuracy
    results['task_accuracy'] = task_accuracy
    results['fail_ratio'] = fail_ratio
    results['metadata'] = {
        'model_name': model_name,
        'prompt_path': prompt_path,
        'problems_dir': problems_dir,
        'seed': seed,
        'replan': replan,
        'fail_probability': fail_probability,
        'enumerate_initial_state': enumerate_initial_state,
        'job_id': unique_id,
    }
    
    if enumerate_initial_state:
        problem_keys = [k for k in results.keys() if isinstance(results[k], dict) and 'initial_state_enum' in results[k]]
        enumeration_accuracy = sum([results[problem]['initial_state_enum']['statistics']['accuracy'] for problem in problem_keys]) / len(problem_keys) if len(problem_keys) > 0 else None
        results['enumeration_accuracy'] = enumeration_accuracy
        if enumeration_accuracy is not None:
            logger.info(f"Enumeration average accuracy: {enumeration_accuracy:.2f}")
        else:
            logger.info("No problems were enumerated")
        
        predicate_enumeration_accuracy = {}
        for problem in problem_keys:
            predicate_enum_stats = results[problem]['initial_state_enum']['statistics']['predicates']
            for predicate in predicate_enum_stats:
                if predicate not in predicate_enumeration_accuracy:
                    predicate_enumeration_accuracy[predicate] = []
                predicate_enumeration_accuracy[predicate].append(predicate_enum_stats[predicate]['accuracy'])
                        
        for predicate in predicate_enumeration_accuracy:
            avg_accuracy = sum(predicate_enumeration_accuracy[predicate]) / len(predicate_enumeration_accuracy[predicate]) if len(predicate_enumeration_accuracy[predicate]) > 0 else None
            if avg_accuracy is not None:
                logger.info(f"Predicate {predicate} average enumeration accuracy: {avg_accuracy:.2f}")
            else:
                logger.info(f"Predicate {predicate} had no enumeration accuracy")
            
    if output_dir is None:
        output_dir = os.curdir
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_{unique_id}.json")
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(results, f)
        
if __name__ == '__main__':
    fire.Fire(main)