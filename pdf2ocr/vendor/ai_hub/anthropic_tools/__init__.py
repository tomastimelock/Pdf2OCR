# Filepath: code_migration/ai_providers/anthropic_tools/__init__.py
# Description: Anthropic Tool Use module - enables Claude to call functions and use tools
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/tools/

"""
Anthropic Tool Use Provider
============================

Enable Claude to use tools and call functions for enhanced capabilities.

Features:
---------
- Define custom tools with JSON Schema
- Force specific tools or let Claude decide
- Auto-execute tool calls with handlers
- Parallel tool execution
- Structured outputs with strict mode
- Server-side tools (web_search, web_fetch, etc.)

Quick Start:
------------
```python
from anthropic_tools import AnthropicToolsProvider

# Initialize
provider = AnthropicToolsProvider()

# Define a tool
provider.register_tool(
    name="get_weather",
    description="Get current weather for a location",
    input_schema={
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "City and state"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
        },
        "required": ["location"]
    }
)

# Call with tools
result = provider.call_with_tools(
    prompt="What's the weather in Paris?",
    tool_choice="auto"
)

# Access tool calls
for tool_call in result["tool_calls"]:
    print(f"Tool: {tool_call['name']}")
    print(f"Input: {tool_call['input']}")
```

Auto-Execute with Handlers:
----------------------------
```python
# Define handler function
def get_weather_handler(location: str, unit: str = "celsius"):
    # Your implementation
    return {"temp": 22, "unit": unit, "condition": "sunny"}

# Register with handler
provider.register_tool(
    name="get_weather",
    description="Get weather",
    input_schema={...},
    handler=get_weather_handler
)

# Auto-execute
result = provider.call_with_tools(
    prompt="What's the weather in Paris?",
    auto_execute=True,
    max_iterations=10
)

# Final text includes tool results
print(result["final_text"])
```

Tool Choice Options:
--------------------
- "auto" - Let Claude decide (default)
- "any" - Claude must use a tool
- "none" - Disable tools
- "tool_name" - Force specific tool
- {"type": "tool", "name": "tool_name"} - Force specific tool (explicit)

Load from File:
---------------
```python
# tools.json:
# [
#   {
#     "name": "search_db",
#     "description": "Search database",
#     "input_schema": {...}
#   }
# ]

provider.load_tools_from_file("tools.json")
```

Strict Mode:
------------
```python
# Guarantee schema compliance
provider.register_tool(
    name="extract_data",
    description="Extract structured data",
    input_schema={...},
    strict=True  # Claude must follow schema exactly
)
```

Simple Tool Call:
-----------------
```python
# Quick one-off tool call
result = provider.simple_tool_call(
    prompt="Extract the name and age",
    tool_name="extract_person",
    tool_description="Extract person info",
    tool_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }
)

print(result["input"])  # {"name": "John", "age": 30}
```

Command Execution:
------------------
```python
commands = '''
TOOL_LOAD file="tools.json"
TOOL_CALL prompt="Get weather in Paris" tool_choice="auto"
TOOL_CALL prompt="Search for restaurants" -> results
SAVE content="${results}" file="output.json"
'''

result = provider.execute_commands(commands)
```

Response Structure:
-------------------
```python
{
    "response": <Response object>,
    "tool_calls": [
        {
            "id": "toolu_01A09q90qw90lq917835",
            "name": "get_weather",
            "input": {"location": "Paris", "unit": "celsius"}
        }
    ],
    "tool_results": [
        {
            "type": "tool_result",
            "tool_use_id": "toolu_01A09q90qw90lq917835",
            "content": '{"temp": 22, "condition": "sunny"}'
        }
    ],
    "final_text": "The weather in Paris is 22°C and sunny.",
    "iterations": 2
}
```

Parallel Tools:
---------------
Claude can call multiple tools in one response:
```python
result = provider.call_with_tools(
    prompt="Get weather in Paris and New York",
)

# result["tool_calls"] will have 2 items
for tc in result["tool_calls"]:
    print(f"{tc['name']}: {tc['input']}")
```

Server Tools:
-------------
Use Anthropic-hosted tools (no handler needed):
```python
# These run on Anthropic's servers
provider.register_tool(
    name="web_search",
    description="Search the web",
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
)
```

Available server tools:
- web_search - Search the internet
- web_fetch - Fetch URL content
- text_editor - Edit text files
- bash - Execute shell commands
- computer - Computer use capabilities

Error Handling:
---------------
```python
def risky_handler(arg: str):
    if not arg:
        raise ValueError("Argument required")
    return process(arg)

provider.register_tool(..., handler=risky_handler)

result = provider.call_with_tools(..., auto_execute=True)

# Check tool_results for errors
for tr in result["tool_results"]:
    if tr.get("is_error"):
        print(f"Error: {tr['content']}")
```

Models Supporting Tools:
------------------------
- claude-sonnet-4-5-20250929 (recommended)
- claude-opus-4-5-20250918
- claude-haiku-4-5-20251001
- All Claude 4.x and 3.x models

Best Practices:
---------------
1. Write detailed descriptions - helps Claude choose correctly
2. Use strict mode for guaranteed schema compliance
3. Handle errors gracefully in tool handlers
4. Set max_iterations to prevent infinite loops
5. Use tool_choice wisely - "auto" for most cases
6. Validate tool inputs in handlers
7. Return JSON-serializable results
8. Keep tool names descriptive and clear

Command Reference:
------------------
TOOL_REGISTER name="..." description="..." schema_file="schema.json"
TOOL_LOAD file="tools.json"
TOOL_CALL prompt="..." [tool_choice="auto"] [output="result.json"]
TOOL_CLEAR
TOOL_LIST
SET var="value"
WAIT seconds=N
PRINT message="..."
SAVE content="..." file="..."

Examples in examples.py

Dependencies:
-------------
- anthropic >= 0.25.0
- python-dotenv

Environment Variables:
----------------------
ANTHROPIC_API_KEY - Your API key (required)
ANTHROPIC_MODEL - Default model (optional, defaults to claude-sonnet-4-5-20250929)

Notes:
------
- Tool names must match: ^[a-zA-Z0-9_-]{1,64}$
- Max 128 tools per request
- Tools consume context window space
- First request with schema has compilation latency
- Schemas cached for 24 hours
"""

from .provider import (
    AnthropicToolsProvider,
    EXAMPLE_TOOLS
)

from .schemas import (
    ToolDefinition,
    ToolCall,
    ToolResult,
    create_tool_definition,
    create_tool_schema,
    validate_tool_name,
    COMMON_TOOL_SCHEMAS
)

from .model_config import (
    TOOL_CONFIG,
    TOOL_SUPPORTED_MODELS,
    get_tool_choice,
    supports_tools
)

__all__ = [
    "AnthropicToolsProvider",
    "EXAMPLE_TOOLS",
    "ToolDefinition",
    "ToolCall",
    "ToolResult",
    "create_tool_definition",
    "create_tool_schema",
    "validate_tool_name",
    "COMMON_TOOL_SCHEMAS",
    "TOOL_CONFIG",
    "TOOL_SUPPORTED_MODELS",
    "get_tool_choice",
    "supports_tools",
]

__version__ = "1.0.0"
