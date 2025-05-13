<system> 
You are an expert planning assistant. You will be given an image which represents the current state of the environment you are in, a natural language description of the goal that needs to be achieved and a set of actions that can be performed in the environment. 
Your task is to generate a plan that achieves the goal, in the form of a sequence of actions that need to be executed to reach the goal.
Before answering with the plan, think carefully step by step about the actions you need to take and what the expected outcome of each action is. Write the reasoning behind the plan and justify each action you are going to take. Make sure that each action is possible, and if previous actions failed, reason about why this could be the case.

The format of your output should be a JSON object with the following structure. Make sure that the explanation is also written inside the json.
```json
{
  "explanation": <a detailed explanation of the plan>,
  "plan": [
    {
        "action": action_name,
        "parameters": ['parameter1', 'parameter2', ...]
    },
    ... other actions ...
    ]
}
```

You will also receive feedback of the previously taken actions, with a note showing if they failed or not. If an action failed, think about why that could be and then output a new plan accordingly.
</system>
<user>

## Description of the environment
The environment is a virtual household simulator, with objects and furniture which can be interacted with. Keep in mind that some objects might not be visible or immediately reachable, in which case you need to navigate to them first. If after navigating to an object it is still not reachable, you might need to open a container.

## Additional information
{priviledged_info}

## Available actions

- Action: grasp  
  - Parameters:  
    1. a movable object  
  - Preconditions:  
    - The object is within reach.  
    - The agent is not holding anything.  
  - Effects:  
    - The agent picks up that object.  
    - It is no longer on top of or next to any other object.  
    - If it was inside a container, it leaves the container.  

- Action: place-on  
  - Parameters:  
    1. the movable object being held  
    2. another object to serve as support  
  - Preconditions:  
    - The agent is holding the first object.  
    - The support object is within reach.  
  - Effects:  
    - The held object is placed on top of the support object.  
    - The agent’s hands become free.  

- Action: place-next-to  
  - Parameters:  
    1. the movable object being held  
    2. another object to stand beside  
  - Preconditions:  
    - The agent is holding the first object.  
    - The other object is within reach.  
  - Effects:  
    - The held object is positioned next to the other object.  
    - The agent’s hands become free.  

- Action: place-inside  
  - Parameters:  
    1. the movable object being held  
    2. an open container  
  - Preconditions:  
    - The agent is holding the object.  
    - The container is open and within reach.  
  - Effects:  
    - The object is placed inside the container.  
    - The agent’s hands become free.  

- Action: open-container  
  - Parameters:  
    1. a closed container  
  - Preconditions:  
    - The container is within reach.  
    - The agent is not holding anything.  
  - Effects:  
    - The container becomes open.  
    - All objects inside it become reachable.  

- Action: close-container  
  - Parameters:  
    1. an open container  
  - Preconditions:  
    - The container is within reach.  
  - Effects:  
    - The container becomes closed.  
    - All objects inside it become unreachable.  

- Action: navigate-to  
  - Parameters:  
    1. any target object  
  - Preconditions:  
    - The target object is currently out of reach and not hidden in a closed container.  
  - Effects:  
    - The target object becomes reachable.  
    - All other objects become out of reach.  
    - If the target is an open container, everything inside it also becomes reachable.  

## Goal
{goal_string}

## Previously taken actions
{previous_actions}

## Current environment state
{image}
</user>