# Agent101

A comprehensive example implementation of a single ReAct (Reasoning + Acting) agent built with Claude AI, demonstrating a modular framework for building intelligent customer support agents.

## Overview

This repository contains `single_react_agent_example.py`, a fully-featured implementation of an AI agent that uses the ReAct pattern to handle customer support queries. The agent can understand customer requests, plan execution steps, use tools to gather information, and synthesize helpful responses.

## Features

### Modular Architecture
The implementation is organized into distinct modules for better maintainability and extensibility:

- **SystemMessage**: Manages system prompts for the agent
- **MemoryBank**: Handles conversation history and memory
- **Toolkit**: Manages tool definitions and execution
- **LLMClient**: Handles LLM API interactions with error handling
- **ReActAgent**: Main agent implementation with state machine

### Key Capabilities

1. **Goal Understanding**: Uses LLM to extract structured information from customer messages
2. **Intelligent Planning**: Creates execution plans for complex requests
3. **Tool Execution**: Executes tools with automatic retry and error recovery
4. **ReAct Loop**: Implements reasoning and acting cycles for dynamic problem-solving
5. **Response Verification**: Validates that responses fully address customer goals
6. **Error Handling**: Robust error handling with fallback strategies

### Built-in Tools

The agent comes with three mock customer support tools:

- `get_order`: Retrieves order information from the database
- `track_shipment`: Gets real-time tracking information for packages
- `process_refund`: Initiates refunds for orders

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jiandai/agent101.git
cd agent101
```

2. Install dependencies:
```bash
pip install anthropic python-dotenv
```

3. Set up your environment variables:
```bash
# Create a .env file with your Anthropic API key
echo "ANTHROPIC_CLAUDE_API_KEY=your_api_key_here" > .env
```

## Usage

Run the example:

```bash
python single_react_agent_example.py
```

### Using the Agent in Your Code

```python
from single_react_agent_example import ReActAgent

# Create an agent instance
agent = ReActAgent()

# Process a customer request
response = agent.process_request(
    "Where is my order ORD-12345? I need to know when it will arrive."
)

print(response)

# View API call summary
agent.show_api_call_summary()
```

### Customizing the Agent

You can customize the agent by providing your own modules:

```python
from single_react_agent_example import (
    ReActAgent,
    LLMClient,
    Toolkit,
    MemoryBank,
    SystemMessage
)

# Create custom components
custom_toolkit = Toolkit()
custom_toolkit.register_tool({
    "name": "custom_tool",
    "description": "Your custom tool",
    "input_schema": {
        "type": "object",
        "properties": {
            "param": {"type": "string"}
        }
    }
})

custom_system_message = SystemMessage("Your custom system prompt")

# Create agent with custom components
agent = ReActAgent(
    toolkit=custom_toolkit,
    system_message=custom_system_message
)
```

## Agent State Machine

The agent follows a state machine pattern:

1. **INIT**: Initializing
2. **UNDERSTANDING**: Understanding the customer's goal
3. **PLANNING**: Creating an execution plan
4. **EXECUTING**: Executing the plan (ReAct loop)
5. **SYNTHESIZING**: Creating the final response
6. **VERIFYING**: Verifying goal completion
7. **COMPLETE**: Task completed successfully
8. **FAILED**: Task failed

## How It Works

1. **Goal Extraction**: The agent analyzes the customer message to extract structured information about what the customer needs
2. **Planning**: For complex requests, the agent creates a step-by-step execution plan
3. **Execution**: The agent executes the plan using a ReAct loop:
   - Reasons about what to do next
   - Uses tools to gather information
   - Processes tool results
   - Synthesizes a response
4. **Verification**: The agent verifies that the final response fully addresses the customer's goal
5. **Retry**: If verification fails, the agent continues the ReAct loop to gather more information

## GitHub Actions Integration

This repository includes a GitHub Actions workflow (`.github/workflows/claude.yml`) that integrates Claude Code for automated assistance:

- Responds to `@claude` mentions in issues and pull requests
- Automatically processes requests and provides help
- Has permissions to read/write code, issues, and pull requests

## Contributing

Feel free to fork this repository and experiment with different:
- System prompts
- Tool definitions
- Planning strategies
- Verification methods

## License

This is an educational example project. Feel free to use and modify as needed.

## Requirements

- Python 3.7+
- anthropic
- python-dotenv

## Environment Variables

- `ANTHROPIC_CLAUDE_API_KEY`: Your Anthropic API key (required)

## Example Queries

Try these example queries:

```python
# Order status inquiry
agent.process_request("Where is my order ORD-12345?")

# Tracking inquiry
agent.process_request("Can you track my package TRK-98765?")

# Refund request
agent.process_request("I need to return order ORD-12345")

# Complex multi-step request
agent.process_request("I ordered something with order ORD-12345, can you tell me where it is and when it will arrive?")
```

## Architecture Highlights

- **Modular Design**: Each component is independent and testable
- **Error Recovery**: Automatic retry with exponential backoff
- **Fallback Strategies**: Regex-based fallbacks when LLM parsing fails
- **Plan Validation**: Validates execution plans before running them
- **Goal Verification**: Uses LLM to verify response completeness

## Learn More

This implementation demonstrates production-grade patterns for building LLM agents:
- State machine architecture
- Tool use and execution
- Error handling and recovery
- Memory management
- Response verification

Perfect for learning about AI agents, ReAct patterns, and building production-ready AI systems.
