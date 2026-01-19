# Filepath: code_migration/ai_providers/anthropic_tools/schemas.py
# Description: Tool definition schemas and helpers for Anthropic Tool Use
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/tools/

"""
Tool definition schemas and helper functions for creating tools.
"""

import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


@dataclass
class ToolDefinition:
    """Definition of a tool for Claude."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    strict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        result = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema
        }
        if self.strict:
            result["strict"] = True
        return result


@dataclass
class ToolCall:
    """Represents a tool call from Claude."""
    id: str
    name: str
    input: Dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of executing a tool."""
    tool_use_id: str
    content: str
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to API format."""
        result = {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content
        }
        if self.is_error:
            result["is_error"] = True
        return result


def validate_tool_name(name: str) -> None:
    """
    Validate tool name matches Anthropic requirements.

    Tool names must match: ^[a-zA-Z0-9_-]{1,64}$

    Args:
        name: Tool name to validate

    Raises:
        ValueError: If name is invalid
    """
    pattern = r'^[a-zA-Z0-9_-]{1,64}$'
    if not re.match(pattern, name):
        raise ValueError(
            f"Tool name '{name}' is invalid. Must match pattern: {pattern}"
        )


def create_tool_definition(
    name: str,
    description: str,
    properties: Dict[str, Any],
    required: Optional[List[str]] = None,
    strict: bool = False
) -> Dict[str, Any]:
    """
    Create a tool definition with proper schema structure.

    Args:
        name: Tool name
        description: What the tool does
        properties: Dictionary of property definitions
        required: List of required property names
        strict: Enable strict mode for guaranteed schema compliance

    Returns:
        Complete tool definition

    Example:
        ```python
        tool = create_tool_definition(
            name="get_weather",
            description="Get current weather",
            properties={
                "location": {
                    "type": "string",
                    "description": "City and state"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"]
                }
            },
            required=["location"]
        )
        ```
    """
    validate_tool_name(name)

    schema = {
        "type": "object",
        "properties": properties
    }

    if required:
        schema["required"] = required

    tool_def = {
        "name": name,
        "description": description,
        "input_schema": schema
    }

    if strict:
        tool_def["strict"] = True

    return tool_def


def create_tool_schema(
    properties: Dict[str, Any],
    required: Optional[List[str]] = None,
    additional_properties: bool = False
) -> Dict[str, Any]:
    """
    Create a JSON Schema for tool input.

    Args:
        properties: Dictionary of property definitions
        required: List of required property names
        additional_properties: Allow additional properties not in schema

    Returns:
        JSON Schema object
    """
    schema = {
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties
    }

    if required:
        schema["required"] = required

    return schema


# Common tool schemas for reuse
COMMON_TOOL_SCHEMAS = {
    "search": {
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
    },

    "get_by_id": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Resource ID"
            }
        },
        "required": ["id"]
    },

    "create_record": {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "Record data"
            }
        },
        "required": ["data"]
    },

    "update_record": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Record ID"
            },
            "data": {
                "type": "object",
                "description": "Updated data"
            }
        },
        "required": ["id", "data"]
    },

    "delete_record": {
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "Record ID"
            }
        },
        "required": ["id"]
    },

    "file_operation": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path"
            },
            "content": {
                "type": "string",
                "description": "File content"
            }
        },
        "required": ["path"]
    },

    "api_call": {
        "type": "object",
        "properties": {
            "endpoint": {
                "type": "string",
                "description": "API endpoint path"
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"],
                "description": "HTTP method"
            },
            "body": {
                "type": "object",
                "description": "Request body"
            },
            "headers": {
                "type": "object",
                "description": "Request headers"
            }
        },
        "required": ["endpoint", "method"]
    },

    "web_search": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            }
        },
        "required": ["query"]
    },

    "web_fetch": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to fetch"
            }
        },
        "required": ["url"]
    },

    "extract_data": {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "Text to extract from"
            },
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Fields to extract"
            }
        },
        "required": ["text", "fields"]
    }
}


# Example property definitions for common types
PROPERTY_EXAMPLES = {
    "string": {
        "type": "string",
        "description": "A text value"
    },

    "integer": {
        "type": "integer",
        "description": "A whole number"
    },

    "number": {
        "type": "number",
        "description": "A numeric value"
    },

    "boolean": {
        "type": "boolean",
        "description": "True or false"
    },

    "array": {
        "type": "array",
        "items": {"type": "string"},
        "description": "List of values"
    },

    "object": {
        "type": "object",
        "description": "Nested object"
    },

    "enum": {
        "type": "string",
        "enum": ["option1", "option2", "option3"],
        "description": "One of predefined values"
    },

    "email": {
        "type": "string",
        "format": "email",
        "description": "Email address"
    },

    "uri": {
        "type": "string",
        "format": "uri",
        "description": "URI/URL"
    },

    "date": {
        "type": "string",
        "format": "date",
        "description": "Date in YYYY-MM-DD format"
    },

    "datetime": {
        "type": "string",
        "format": "date-time",
        "description": "ISO 8601 datetime"
    },

    "uuid": {
        "type": "string",
        "format": "uuid",
        "description": "UUID identifier"
    }
}


def create_extraction_tool(
    name: str,
    description: str,
    output_schema: Dict[str, Any],
    strict: bool = True
) -> Dict[str, Any]:
    """
    Create a tool for structured data extraction.

    Args:
        name: Tool name
        description: What to extract
        output_schema: Schema for extracted data
        strict: Enable strict mode (recommended for extraction)

    Returns:
        Tool definition optimized for extraction

    Example:
        ```python
        tool = create_extraction_tool(
            name="extract_person",
            description="Extract person information",
            output_schema={
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "email": {"type": "string", "format": "email"}
            }
        )
        ```
    """
    return create_tool_definition(
        name=name,
        description=description,
        properties=output_schema,
        required=list(output_schema.keys()),
        strict=strict
    )


def create_server_tool(tool_type: str) -> Dict[str, Any]:
    """
    Create a definition for Anthropic server-side tools.

    Args:
        tool_type: One of: web_search, web_fetch, text_editor, bash, computer

    Returns:
        Tool definition for server tool
    """
    server_tools = {
        "web_search": {
            "name": "web_search",
            "description": "Search the internet for information",
            "input_schema": COMMON_TOOL_SCHEMAS["web_search"]
        },
        "web_fetch": {
            "name": "web_fetch",
            "description": "Fetch content from a URL",
            "input_schema": COMMON_TOOL_SCHEMAS["web_fetch"]
        },
        "text_editor": {
            "name": "text_editor",
            "description": "Edit text files",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                        "description": "Editor command"
                    },
                    "path": {
                        "type": "string",
                        "description": "File path"
                    }
                },
                "required": ["command", "path"]
            }
        },
        "bash": {
            "name": "bash",
            "description": "Execute bash commands",
            "input_schema": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Bash command to execute"
                    }
                },
                "required": ["command"]
            }
        },
        "computer": {
            "name": "computer",
            "description": "Computer use capabilities",
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["key", "type", "mouse_move", "left_click", "screenshot"],
                        "description": "Action to perform"
                    }
                },
                "required": ["action"]
            }
        }
    }

    if tool_type not in server_tools:
        raise ValueError(f"Unknown server tool: {tool_type}")

    return server_tools[tool_type]
