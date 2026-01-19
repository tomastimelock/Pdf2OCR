# Anthropic Tool Use Provider

Complete standalone module for Claude function calling and tool use capabilities.

## Features

- **Tool Registration** - Define custom tools with JSON Schema
- **Auto-Execute** - Automatically run tool handlers and continue conversation
- **Parallel Tools** - Claude can call multiple tools simultaneously
- **Strict Mode** - Guarantee schema compliance for structured outputs
- **Server Tools** - Use Anthropic-hosted tools (web_search, web_fetch, etc.)
- **Command Execution** - Run workflows with declarative commands
- **Error Handling** - Graceful error handling in tool execution

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

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

# Access results
for tool_call in result["tool_calls"]:
    print(f"Tool: {tool_call['name']}")
    print(f"Input: {tool_call['input']}")
```

## Auto-Execute with Handlers

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

print(result["final_text"])  # Includes tool results
```

## Tool Choice Options

- `"auto"` - Let Claude decide (default)
- `"any"` - Claude must use a tool
- `"none"` - Disable tools
- `"tool_name"` - Force specific tool
- `{"type": "tool", "name": "tool_name"}` - Explicit format

## Structured Extraction

```python
from anthropic_tools import create_extraction_tool

tool = create_extraction_tool(
    name="extract_person",
    description="Extract person information",
    output_schema={
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "email": {"type": "string", "format": "email"}
    },
    strict=True  # Guarantee schema compliance
)

result = provider.simple_tool_call(
    prompt="John Smith is 35, email: john@example.com",
    **tool
)

print(result["input"])  # {"name": "John Smith", "age": 35, "email": "john@example.com"}
```

## Load Tools from File

```python
# tools.json
[
  {
    "name": "search_db",
    "description": "Search database",
    "input_schema": {
      "type": "object",
      "properties": {
        "query": {"type": "string"}
      },
      "required": ["query"]
    }
  }
]

# Load
provider.load_tools_from_file("tools.json")
```

## Response Structure

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

## Command Execution

```python
commands = '''
TOOL_LOAD file="tools.json"
TOOL_CALL prompt="Get weather in Paris" tool_choice="auto"
TOOL_CALL prompt="Search for restaurants" -> results
SAVE content="${results}" file="output.json"
'''

result = provider.execute_commands(commands)
```

### Available Commands

- `TOOL_REGISTER name="..." description="..." schema_file="schema.json"`
- `TOOL_LOAD file="tools.json"`
- `TOOL_CALL prompt="..." [tool_choice="auto"] [output="result.json"]`
- `TOOL_CLEAR` - Clear all registered tools
- `TOOL_LIST` - List registered tools
- `SET var="value"` - Set variable
- `WAIT seconds=N` - Wait N seconds
- `PRINT message="..."` - Print message
- `SAVE content="..." file="..."` - Save to file

## Server Tools

Use Anthropic-hosted tools (no handler needed):

```python
from anthropic_tools import create_server_tool

# Web search
provider.register_tool(**create_server_tool("web_search"))

# Web fetch
provider.register_tool(**create_server_tool("web_fetch"))
```

Available server tools:
- `web_search` - Search the internet
- `web_fetch` - Fetch URL content
- `text_editor` - Edit text files
- `bash` - Execute shell commands
- `computer` - Computer use capabilities

## Helper Functions

```python
from anthropic_tools import (
    create_tool_definition,
    create_tool_schema,
    create_extraction_tool,
    validate_tool_name,
    COMMON_TOOL_SCHEMAS
)

# Create tool definition
tool = create_tool_definition(
    name="my_tool",
    description="What it does",
    properties={
        "param1": {"type": "string"},
        "param2": {"type": "integer"}
    },
    required=["param1"]
)

# Use common schema
search_schema = COMMON_TOOL_SCHEMAS["search"]

# Validate tool name
validate_tool_name("my_tool")  # Raises if invalid
```

## Supported Models

- `claude-sonnet-4-5-20250929` (recommended)
- `claude-opus-4-5-20250918`
- `claude-haiku-4-5-20251001`
- All Claude 4.x and 3.x models

## Environment Variables

```bash
ANTHROPIC_API_KEY=your_api_key_here
ANTHROPIC_MODEL=claude-sonnet-4-5-20250929  # Optional
```

## Examples

See `examples.py` for comprehensive usage examples:

```bash
python examples.py
```

## Best Practices

1. **Write Detailed Descriptions** - Helps Claude choose correctly
2. **Use Strict Mode** - For guaranteed schema compliance
3. **Handle Errors** - Gracefully handle tool execution errors
4. **Set Max Iterations** - Prevent infinite loops in auto-execute
5. **Use Tool Choice Wisely** - "auto" for most cases
6. **Validate Inputs** - In your tool handlers
7. **Return JSON** - Keep results JSON-serializable
8. **Clear Tool Names** - Descriptive and specific

## Common Tool Schemas

The module includes common schemas you can reuse:

- `search` - Search with query and limit
- `get_by_id` - Get resource by ID
- `create_record` - Create with data
- `update_record` - Update by ID
- `delete_record` - Delete by ID
- `file_operation` - File path and content
- `api_call` - REST API calls
- `web_search` - Web search
- `web_fetch` - Fetch URL
- `extract_data` - Data extraction

## Tool Name Rules

Tool names must match: `^[a-zA-Z0-9_-]{1,64}$`

- Only alphanumeric, underscore, hyphen
- 1-64 characters
- No spaces or special characters

## Limitations

- Max 128 tools per request
- Tool definitions consume context window
- First request with schema has compilation latency (24hr cache)
- Strict mode incompatible with citations and message prefilling

## Error Handling

```python
def risky_handler(arg: str):
    if not arg:
        raise ValueError("Argument required")
    return process(arg)

provider.register_tool(..., handler=risky_handler)

result = provider.call_with_tools(..., auto_execute=True)

# Check for errors
for tr in result["tool_results"]:
    if tr.get("is_error"):
        print(f"Error: {tr['content']}")
```

## Files

- `__init__.py` - Module exports and documentation
- `provider.py` - Main AnthropicToolsProvider class
- `schemas.py` - Tool definition schemas and helpers
- `model_config.py` - Model configuration and helpers
- `examples.py` - Usage examples
- `requirements.txt` - Dependencies
- `README.md` - This file

## License

Copy-paste ready, self-contained module for DocFlow pipeline system.
