# A Systematic Risk Regulation Framework

This repository contains the implementation of a multi-agent system, where each agent is represented by an LLM-based model. The agents interact with each other through a dynamic environment, guided by a central regulator (LLMPlanner). The parameters for each agent, such as risk preferences, wealth, and other characteristics, are configurable in JSON and YAML files.

## File Overview

### `LLMAgent.py`
This file defines the behavior and attributes of the agents. Each agent interacts with the environment, makes decisions based on its internal model, and follows the instructions from the central regulator.

### `LLMPlanner.py`
The regulator module that supervises the decision-making process of all agents. It ensures that agents adhere to certain guidelines or policies and can modify the environment or agent behavior accordingly.

### `env.py`
This file defines the interactive environment in which the agents operate. It controls the state of the environment and provides feedback to the agents based on their actions.

### Configuration Files

- **`config.json`**: This JSON file defines agent-specific parameters, such as risk preferences, wealth, and other customizable attributes.
- **`config.yaml`**: A YAML file that serves as an alternative configuration format for defining agent parameters and other system settings.

## How to Use

1. **Setup**: Clone the repository to your local machine.
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   cd repository-name
