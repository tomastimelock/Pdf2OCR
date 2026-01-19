# Filepath: code_migration/ai_providers/anthropic_tools/provider.py
# Description: AnthropicToolsProvider - Claude function calling and tool use
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/tools/provider.py

"""
Anthropic Tools Provider
Uses the Anthropic Messages API with tool use capabilities for function calling.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Callable
from anthropic import Anthropic
from dotenv import load_dotenv

from .model_config import (
    get_default_model, get_tool_choice, TOOL_CONFIG
)
from .schemas import ToolDefinition, ToolCall, ToolResult, validate_tool_name


class AnthropicToolsProvider:
    """Provider for Anthropic tool use with function calling capabilities."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Anthropic Tools Provider.

        Args:
            api_key: Anthropic API key. If not provided, loads from ANTHROPIC_API_KEY env var.
            model: Model to use. If not provided, loads from ANTHROPIC_MODEL env var or defaults.
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable")

        self.model = model or os.getenv("ANTHROPIC_MODEL", get_default_model())
        self.client = Anthropic(api_key=self.api_key)
        self.tools: List[Dict[str, Any]] = []
        self.tool_handlers: Dict[str, Callable] = {}

    def _extract_text(self, response) -> str:
        """Extract text content from a response object."""
        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return ''.join(text_parts)

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Optional[Callable] = None,
        strict: bool = False
    ) -> None:
        """
        Register a tool for Claude to use.

        Args:
            name: Tool name (must match ^[a-zA-Z0-9_-]{1,64}$)
            description: Description of what the tool does
            input_schema: JSON Schema for tool inputs
            handler: Optional Python function to handle the tool call
            strict: Enable strict schema validation (default: False)
        """
        # Validate tool name
        validate_tool_name(name)

        tool_def = {
            "name": name,
            "description": description,
            "input_schema": input_schema
        }

        if strict:
            tool_def["strict"] = True

        self.tools.append(tool_def)

        if handler:
            self.tool_handlers[name] = handler

    def clear_tools(self) -> None:
        """Clear all registered tools."""
        self.tools = []
        self.tool_handlers = {}

    def load_tools_from_file(self, file_path: str) -> None:
        """
        Load tool definitions from a JSON file.

        Args:
            file_path: Path to JSON file containing tool definitions
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Handle both single tool and array of tools
        tools = data if isinstance(data, list) else [data]

        for tool in tools:
            self.tools.append({
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool.get("input_schema", tool.get("schema", {})),
                **({"strict": True} if tool.get("strict") else {})
            })

    def call_with_tools(
        self,
        prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Union[str, Dict[str, Any]] = "auto",
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        auto_execute: bool = False,
        max_iterations: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Call Claude with tool use capabilities.

        Args:
            prompt: User prompt
            tools: Tool definitions (uses registered tools if not provided)
            tool_choice: How Claude should use tools:
                - "auto": Claude decides (default)
                - "any": Must use a tool
                - "none": No tools
                - {"type": "tool", "name": "tool_name"}: Specific tool
            model: Model to use
            system: System prompt
            max_tokens: Maximum tokens
            auto_execute: Automatically execute tools and continue conversation
            max_iterations: Max tool use iterations when auto_execute is True
            **kwargs: Additional parameters

        Returns:
            Dictionary with response, tool_calls, and final_text
        """
        used_model = model or self.model
        tools_to_use = tools or self.tools

        if not tools_to_use:
            raise ValueError("No tools provided. Register tools or pass them directly.")

        # Build tool_choice parameter
        tc = get_tool_choice(tool_choice)

        messages = [{"role": "user", "content": prompt}]

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "messages": messages,
            "tools": tools_to_use,
            "tool_choice": tc
        }

        if system:
            params["system"] = system

        params.update(kwargs)

        response = self.client.messages.create(**params)

        result = {
            "response": response,
            "tool_calls": [],
            "tool_results": [],
            "final_text": "",
            "iterations": 0
        }

        # Process response
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            result["iterations"] = iteration

            # Extract tool calls from response
            tool_calls = []
            text_parts = []

            for block in response.content:
                if block.type == "tool_use":
                    tool_calls.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input
                    })
                elif block.type == "text":
                    text_parts.append(block.text)

            result["tool_calls"].extend(tool_calls)

            if text_parts:
                result["final_text"] = ''.join(text_parts)

            # If no tool calls or not auto-executing, we're done
            if not tool_calls or not auto_execute:
                break

            # Execute tools and continue
            tool_results = []
            for tc_item in tool_calls:
                tool_name = tc_item["name"]
                tool_input = tc_item["input"]

                if tool_name in self.tool_handlers:
                    try:
                        tool_result = self.tool_handlers[tool_name](**tool_input)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc_item["id"],
                            "content": json.dumps(tool_result) if not isinstance(tool_result, str) else tool_result
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc_item["id"],
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        })
                else:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc_item["id"],
                        "content": f"Tool handler not registered for: {tool_name}",
                        "is_error": True
                    })

            result["tool_results"].extend(tool_results)

            # Continue conversation with tool results
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            params["messages"] = messages
            response = self.client.messages.create(**params)
            result["response"] = response

            # Check if we should continue (more tool use)
            if response.stop_reason != "tool_use":
                # Final response - extract text
                for block in response.content:
                    if block.type == "text":
                        result["final_text"] = block.text
                break

        return result

    def simple_tool_call(
        self,
        prompt: str,
        tool_name: str,
        tool_description: str,
        tool_schema: Dict[str, Any],
        model: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a simple tool call with a single tool.

        Args:
            prompt: User prompt
            tool_name: Name of the tool
            tool_description: Description of the tool
            tool_schema: JSON Schema for tool inputs
            model: Model to use
            **kwargs: Additional parameters

        Returns:
            Tool call result with name and input
        """
        tool = {
            "name": tool_name,
            "description": tool_description,
            "input_schema": tool_schema
        }

        result = self.call_with_tools(
            prompt=prompt,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},
            model=model,
            **kwargs
        )

        if result["tool_calls"]:
            return result["tool_calls"][0]
        return {"error": "No tool call made"}

    def execute_commands(self, command_string: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute a workflow of tool operations.

        Command Format:
        ---------------
        TOOL_REGISTER name="..." description="..." schema_file="schema.json"
        TOOL_CALL prompt="..." [tool_choice="auto"] [output="result.json"]
        TOOL_CALL prompt="..." -> variable_name
        TOOL_LOAD file="tools.json"
        TOOL_CLEAR
        TOOL_LIST

        SET var="value"
        WAIT seconds=N
        PRINT message="..."
        SAVE content="..." file="..."

        Args:
            command_string: Multi-line string containing commands
            verbose: Print execution progress

        Returns:
            Dictionary with execution results and metadata
        """
        variables = {}
        results = []
        last_output = None

        def substitute_vars(value: str) -> str:
            if not isinstance(value, str):
                return value
            if last_output and "${LAST_OUTPUT}" in value:
                value = value.replace("${LAST_OUTPUT}", str(last_output))
            for var_name, var_value in variables.items():
                value = value.replace(f"${{{var_name}}}", str(var_value))
            return value

        def parse_args(arg_string: str) -> Dict[str, Any]:
            args = {}
            pattern = r'(\w+)=(?:"([^"]*)"|(\S+))'
            matches = re.findall(pattern, arg_string)
            for match in matches:
                key = match[0]
                value = match[1] if match[1] else match[2]
                if value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif re.match(r'^\d+\.\d+$', value):
                    value = float(value)
                if isinstance(value, str):
                    value = substitute_vars(value)
                args[key] = value
            return args

        def parse_arrow_assignment(line: str):
            arrow_match = re.search(r'->\s*(\w+)\s*$', line)
            if arrow_match:
                var_name = arrow_match.group(1)
                line = line[:arrow_match.start()].strip()
                return line, var_name
            return line, None

        lines = command_string.strip().split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('//'):
                continue

            try:
                line, assign_var = parse_arrow_assignment(line)
                parts = line.split(None, 1)
                if not parts:
                    continue

                command = parts[0].upper()
                arg_string = parts[1] if len(parts) > 1 else ""
                args = parse_args(arg_string)

                if verbose:
                    print(f"[Line {line_num}] Executing: {command}")

                if command == "SET":
                    for key, value in args.items():
                        variables[key] = value
                        if verbose:
                            print(f"  Set ${{{key}}} = {value}")

                elif command == "WAIT":
                    seconds = args.get('seconds', 1)
                    if verbose:
                        print(f"  Waiting {seconds} seconds...")
                    time.sleep(seconds)

                elif command == "PRINT":
                    message = args.get('message', args.get('text', ''))
                    print(f"  {message}")

                elif command == "SAVE":
                    content = args.get('content', '')
                    file_path = args.get('file', '')
                    if file_path:
                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        if isinstance(content, (dict, list)):
                            content = json.dumps(content, indent=2)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(str(content))
                        if verbose:
                            print(f"  Saved to: {file_path}")

                elif command == "TOOL_REGISTER":
                    name = args.get('name', '')
                    description = args.get('description', '')
                    schema_file = args.get('schema_file', '')

                    with open(schema_file, 'r', encoding='utf-8') as f:
                        schema = json.load(f)

                    self.register_tool(name=name, description=description, input_schema=schema)

                    if verbose:
                        print(f"  [OK] Registered tool: {name}")

                    results.append({
                        'command': 'TOOL_REGISTER',
                        'line': line_num,
                        'result': {'name': name}
                    })

                elif command == "TOOL_LOAD":
                    file_path = args.get('file', '')
                    self.load_tools_from_file(file_path)

                    if verbose:
                        print(f"  [OK] Loaded tools from: {file_path}")
                        print(f"  Tools: {[t['name'] for t in self.tools]}")

                    results.append({
                        'command': 'TOOL_LOAD',
                        'line': line_num,
                        'result': {'file': file_path, 'tools': [t['name'] for t in self.tools]}
                    })

                elif command == "TOOL_CLEAR":
                    self.clear_tools()
                    if verbose:
                        print("  [OK] Cleared all tools")

                    results.append({
                        'command': 'TOOL_CLEAR',
                        'line': line_num,
                        'result': {}
                    })

                elif command == "TOOL_LIST":
                    if verbose:
                        print(f"  Registered tools: {len(self.tools)}")
                        for tool in self.tools:
                            print(f"    - {tool['name']}: {tool['description'][:50]}...")

                    results.append({
                        'command': 'TOOL_LIST',
                        'line': line_num,
                        'result': {'tools': [t['name'] for t in self.tools]}
                    })

                elif command == "TOOL_CALL":
                    prompt = args.get('prompt', '')
                    tool_choice = args.get('tool_choice', 'auto')
                    output = args.get('output', None)
                    auto_execute = args.get('auto_execute', False)

                    call_params = {
                        'prompt': prompt,
                        'tool_choice': tool_choice,
                        'auto_execute': auto_execute
                    }

                    if 'model' in args:
                        call_params['model'] = args['model']
                    if 'system' in args:
                        call_params['system'] = args['system']

                    result_data = self.call_with_tools(**call_params)

                    # Prepare output
                    output_data = {
                        'tool_calls': result_data['tool_calls'],
                        'final_text': result_data['final_text'],
                        'iterations': result_data['iterations']
                    }

                    if assign_var:
                        variables[assign_var] = output_data
                        last_output = output_data
                        if verbose:
                            print(f"  [OK] Stored in ${{{assign_var}}}")
                            if result_data['tool_calls']:
                                print(f"  Tool calls: {[tc['name'] for tc in result_data['tool_calls']]}")
                    elif output:
                        Path(output).parent.mkdir(parents=True, exist_ok=True)
                        with open(output, 'w', encoding='utf-8') as f:
                            json.dump(output_data, f, indent=2)
                        last_output = output
                        if verbose:
                            print(f"  [OK] Saved to: {output}")
                    else:
                        last_output = output_data
                        if verbose:
                            if result_data['tool_calls']:
                                print(f"  Tool calls: {result_data['tool_calls']}")
                            if result_data['final_text']:
                                preview = result_data['final_text'][:100] + "..."
                                print(f"  Response: {preview}")

                    results.append({
                        'command': 'TOOL_CALL',
                        'line': line_num,
                        'result': output_data
                    })

                else:
                    if verbose:
                        print(f"  [WARN] Unknown command: {command}")

            except Exception as e:
                error_msg = f"Error on line {line_num} ({command}): {str(e)}"
                if verbose:
                    print(f"  [ERROR] {error_msg}")
                results.append({
                    'command': command,
                    'line': line_num,
                    'error': str(e)
                })

        return {
            'success': True,
            'results': results,
            'variables': variables,
            'last_output': last_output,
            'total_commands': len([r for r in results if 'error' not in r]),
            'total_errors': len([r for r in results if 'error' in r])
        }


# Example tool definitions
EXAMPLE_TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. 'San Francisco, CA'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "search_database",
        "description": "Search a database for records",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical calculations",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "send_email",
        "description": "Send an email message",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {
                    "type": "string",
                    "description": "Recipient email address"
                },
                "subject": {
                    "type": "string",
                    "description": "Email subject"
                },
                "body": {
                    "type": "string",
                    "description": "Email body"
                }
            },
            "required": ["to", "subject", "body"]
        }
    }
]
