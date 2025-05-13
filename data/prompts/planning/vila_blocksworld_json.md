<system> 
You are an expert planning assistant. You will be given an image which represents the current state of the environment you are in, a natural language description of the goal that needs to be achieved and a set of actions that can be performed in the environment. 
Your task is to generate a plan that achieves the goal, in the form of a sequence of actions that need to be executed to reach the goal.
The format of your output should be a JSON object with the following structure:
```json
{
  "plan": [
    {
        "action": action_name,
        "parameters": {
            parameter_name: parameter_value
        }
    },
    ... other actions ...
    ]
}
```

You will also receive feedback of the previously taken actions, with a note showing if they failed or not. If an action failed, think about why that could be and then output a new plan accordingly.
</system>
<user>
## Description of the environment
The environment is about colored blocks and how they relate to each other. In the environment, the blocks will be arranged in columns, spanning from left to right. Keep in mind that some of these columns can be empty with no blocks currently placed in them. Within a column one or multiple blocks of different colors can be stacked on top of each other. Your task is to correctly evaluate the question based on the image provided.

## Available actions
You have only one action available, called `moveblock(block, column)`. This action allows you to move a block from its current column to the specified column. In order to perform this action, the block you want to move must be the topmost block of its column and must not already be in the target column. If the action is valid, the block will be moved to the specified column and will be placed on top of any blocks that are already in that column, if any.
To refer to the blocks, use lowercase letters for the colors: 'r' for red, 'g' for green, 'b' for blue, 'y' for yellow, 'p' for purple, 'o' for orange. To refer to the columns, use the labels provided in the image, 'c1', 'c2', 'c3', 'c4' and 'c5'.

## Goal
{goal_string}

## Previously taken actions
{previous_actions}

## Current environment state
{image}
</user>