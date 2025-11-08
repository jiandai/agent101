# Agent101

Agent101 is a comprehensive example of a single ReAct (Reasoning + Acting) agent built with Claude AI. It demonstrates how to stitch together planning, tool orchestration, memory, and response validation to deliver production-grade customer support workflows.

## At a Glance

- **Language:** Python 3.7+
- **Key file:** `single_react_agent_example.py`
- **Primary pattern:** ReAct loop with verification and retry
- **Use case:** End-to-end customer support agent (order lookup, shipment tracking, refunds)

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Prerequisites](#prerequisites)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Feature Highlights](#feature-highlights)
8. [Architecture](#architecture)
9. [Extending the Agent](#extending-the-agent)
10. [GitHub Actions Integration](#github-actions-integration)
11. [Troubleshooting](#troubleshooting)
12. [Contributing](#contributing)
13. [License](#license)

## Overview

`single_react_agent_example.py` contains a fully-featured implementation of an AI agent that:

- understands customer goals,
- plans multi-step solutions,
- calls tools with automatic retry,
- synthesizes natural language answers, and
- verifies that a response satisfies the original request before replying.

The example ships with mock customer support tools, making it easy to follow the execution path from request intake to completed response.

## Project Structure

```text
.
├── README.md
├── single_react_agent_example.py   # Complete ReAct implementation and demo entry point
└── .github/
    └── workflows/
        └── claude.yml              # Claude Code GitHub Action
```

Key modules inside `single_react_agent_example.py`:

- `SystemMessage`: Centralizes system prompt management.
- `MemoryBank`: Captures and recalls conversation history.
- `Toolkit`: Registers and dispatches tool implementations.
- `LLMClient`: Wraps the Anthropic API and handles recoverable errors.
- `ReActAgent`: Orchestrates the overall state machine and ReAct loop.

## Prerequisites

- Python 3.7 or newer
- An Anthropic Claude API key
- `pip` for dependency management

## Quick Start

1. **Clone the repository**

   ```bash
   git clone https://github.com/jiandai/agent101.git
   cd agent101
   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install anthropic python-dotenv
   ```

4. **Configure environment variables** (see [Configuration](#configuration))

5. **Run the demo**

   ```bash
   python single_react_agent_example.py
   ```

You should see the agent reason through a sample customer request, call the mock tools, and produce a final response.

## Configuration

Create a `.env` file in the project root with your Anthropic credentials:

```bash
echo "ANTHROPIC_CLAUDE_API_KEY=your_api_key_here" > .env
```

The demo loads this file automatically using `python-dotenv`. If you already manage environment variables through another system, you can skip the `.env` file and export `ANTHROPIC_CLAUDE_API_KEY` in your shell instead.

## Usage

### Command-line demo

```bash
python single_react_agent_example.py
```

### Embedding in your own application

```python
from single_react_agent_example import ReActAgent

agent = ReActAgent()
response = agent.process_request(
    "Where is my order ORD-12345? I need to know when it will arrive."
)

print(response)
agent.show_api_call_summary()
```

### Example queries

```python
# Order status inquiry
agent.process_request("Where is my order ORD-12345?")

# Tracking inquiry
agent.process_request("Can you track my package TRK-98765?")

# Refund request
agent.process_request("I need to return order ORD-12345")

# Multi-step request
agent.process_request(
    "I ordered something with order ORD-12345. Where is it and when will it arrive?"
)
```

## Feature Highlights

- **Goal understanding:** Extracts structured objectives from natural language.
- **Planning:** Builds a step-by-step plan for complex tasks.
- **ReAct loop:** Alternates reasoning and tool use with guardrails on tool chaining.
- **Tool execution:** Built-in retry and error handling for external calls.
- **Result verification:** Validates that responses satisfy the original goal.
- **Memory management:** Maintains conversational context across turns.

Built-in tools:

- `get_order`: Mock order metadata lookup.
- `track_shipment`: Returns simulated tracking events.
- `process_refund`: Initiates a refund flow.

## Architecture

### State machine

The agent advances through the following states:

1. `INIT` – Initial setup
2. `UNDERSTANDING` – Parse the customer goal
3. `PLANNING` – Draft a plan of attack
4. `EXECUTING` – Run the ReAct loop
5. `SYNTHESIZING` – Compose the final response
6. `VERIFYING` – Confirm the response addresses the goal
7. `COMPLETE` – Success path
8. `FAILED` – Escalation path with error reasons

### How the ReAct loop works

1. **Goal extraction:** Parse the intent and constraints from the user message.
2. **Plan creation:** Decide on the tool sequence required to fulfill the request.
3. **Tool reasoning:** For each step, generate a thought and choose the next tool.
4. **Tool execution:** Call the tool, parse the result, and stash it in memory.
5. **Response synthesis:** Produce a customer-facing answer once enough context exists.
6. **Verification:** Sanity-check the answer; if missing information, loop again.

## Extending the Agent

You can supply custom components when instantiating `ReActAgent`:

```python
from single_react_agent_example import (
    ReActAgent,
    LLMClient,
    Toolkit,
    MemoryBank,
    SystemMessage,
)

toolkit = Toolkit()
toolkit.register_tool({
    "name": "custom_tool",
    "description": "Describe the tool's purpose",
    "input_schema": {
        "type": "object",
        "properties": {
            "param": {"type": "string"},
        },
        "required": ["param"],
    },
})

agent = ReActAgent(
    toolkit=toolkit,
    system_message=SystemMessage("Your custom system prompt"),
    memory_bank=MemoryBank(),
    llm_client=LLMClient(),
)
```

Suggested extension ideas:

- Add real integrations (order management, ticketing, CRMs).
- Swap in a different LLM provider through `LLMClient`.
- Experiment with alternative planning or verification prompts.
- Persist `MemoryBank` across sessions to build long-lived agents.

## GitHub Actions Integration

The repository ships with `.github/workflows/claude.yml`, which wires Claude Code into issues and pull requests. Mention `@claude` (or your configured trigger phrase) to have the action:

- Review or describe changes on pull requests.
- Triage or answer questions on issues.
- Operate with permissions to write to issues, PRs, metadata, and contents.

To enable the workflow, add the `CLAUDE_CODE_OAUTH_TOKEN` secret in your GitHub repository settings.

## Troubleshooting

- **Missing API key:** Ensure `ANTHROPIC_CLAUDE_API_KEY` is present in `.env` or exported in your shell.
- **Python version errors:** Recreate the virtual environment with Python 3.7+.
- **Rate limits or API errors:** The sample `LLMClient` retries transient Anthropic errors; permanent failures are surfaced with actionable messages.
- **No output:** Run with `python -m single_react_agent_example` to ensure Python’s module path resolves correctly.

## Contributing

Contributions are welcome! Ideas to explore:

- Try different system prompts or guardrails.
- Add new tools or adapters to real services.
- Improve planning, verification, or memory strategies.
- Write unit tests around specific agent components.

Open an issue or pull request with your proposal and we’ll take a look.

## License

This project is shared for educational use. Adapt and extend it freely for your own prototypes or internal tooling.
