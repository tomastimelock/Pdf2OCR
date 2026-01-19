# Filepath: code_migration/ai_providers/anthropic_tools/examples.py
# Description: Usage examples for Anthropic Tool Use
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/tools/

"""
Usage examples for Anthropic Tool Use provider.
"""

import json
from anthropic_tools import AnthropicToolsProvider, create_tool_definition, create_extraction_tool


def example_basic_tool():
    """Example: Basic tool registration and use."""
    print("\n=== Example: Basic Tool Use ===\n")

    provider = AnthropicToolsProvider()

    # Register a weather tool
    provider.register_tool(
        name="get_weather",
        description="Get current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. 'Paris, France'"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        }
    )

    # Call with the tool
    result = provider.call_with_tools(
        prompt="What's the weather like in Paris?",
        tool_choice="auto"
    )

    print("Tool calls:", json.dumps(result["tool_calls"], indent=2))
    print("\nResponse text:", result["final_text"])


def example_auto_execute():
    """Example: Auto-execute tools with handlers."""
    print("\n=== Example: Auto-Execute with Handlers ===\n")

    provider = AnthropicToolsProvider()

    # Define handler function
    def get_weather_handler(location: str, unit: str = "celsius"):
        """Simulated weather API."""
        weather_data = {
            "Paris, France": {"temp": 18, "condition": "Cloudy", "humidity": 65},
            "New York, NY": {"temp": 22, "condition": "Sunny", "humidity": 55},
            "Tokyo, Japan": {"temp": 25, "condition": "Rainy", "humidity": 80},
        }

        data = weather_data.get(location, {"temp": 20, "condition": "Unknown", "humidity": 50})

        if unit == "fahrenheit":
            data["temp"] = data["temp"] * 9/5 + 32

        return {
            "location": location,
            "temperature": data["temp"],
            "unit": unit,
            "condition": data["condition"],
            "humidity": data["humidity"]
        }

    # Register with handler
    provider.register_tool(
        name="get_weather",
        description="Get current weather for a location",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        },
        handler=get_weather_handler
    )

    # Auto-execute
    result = provider.call_with_tools(
        prompt="What's the weather in Paris and New York?",
        auto_execute=True,
        max_iterations=10
    )

    print("Tool calls made:", len(result["tool_calls"]))
    print("\nFinal response:")
    print(result["final_text"])
    print(f"\nCompleted in {result['iterations']} iterations")


def example_structured_extraction():
    """Example: Structured data extraction with strict mode."""
    print("\n=== Example: Structured Data Extraction ===\n")

    provider = AnthropicToolsProvider()

    # Create extraction tool with strict mode
    extraction_tool = create_extraction_tool(
        name="extract_person",
        description="Extract person information from text",
        output_schema={
            "name": {
                "type": "string",
                "description": "Full name"
            },
            "age": {
                "type": "integer",
                "description": "Age in years"
            },
            "email": {
                "type": "string",
                "format": "email",
                "description": "Email address"
            },
            "city": {
                "type": "string",
                "description": "City of residence"
            }
        },
        strict=True
    )

    # Use simple_tool_call for extraction
    result = provider.simple_tool_call(
        prompt="""
        Extract information from this text:

        John Smith is 35 years old and lives in Seattle.
        You can reach him at john.smith@email.com
        """,
        tool_name=extraction_tool["name"],
        tool_description=extraction_tool["description"],
        tool_schema=extraction_tool["input_schema"]
    )

    print("Extracted data:")
    print(json.dumps(result["input"], indent=2))


def example_multiple_tools():
    """Example: Multiple tools with tool choice."""
    print("\n=== Example: Multiple Tools ===\n")

    provider = AnthropicToolsProvider()

    # Register multiple tools
    tools = [
        {
            "name": "get_weather",
            "description": "Get current weather",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "search_restaurants",
            "description": "Search for restaurants",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "cuisine": {"type": "string"}
                },
                "required": ["location"]
            }
        },
        {
            "name": "book_reservation",
            "description": "Book a restaurant reservation",
            "input_schema": {
                "type": "object",
                "properties": {
                    "restaurant": {"type": "string"},
                    "time": {"type": "string"},
                    "party_size": {"type": "integer"}
                },
                "required": ["restaurant", "time", "party_size"]
            }
        }
    ]

    for tool in tools:
        provider.register_tool(**tool)

    # Let Claude choose
    result = provider.call_with_tools(
        prompt="Find Italian restaurants in Paris",
        tool_choice="auto"
    )

    print("Claude chose to use:")
    for tc in result["tool_calls"]:
        print(f"  - {tc['name']} with {tc['input']}")


def example_parallel_tools():
    """Example: Parallel tool execution."""
    print("\n=== Example: Parallel Tool Calls ===\n")

    provider = AnthropicToolsProvider()

    # Handler for weather
    def get_weather(location: str, unit: str = "celsius"):
        return {"location": location, "temp": 20, "unit": unit}

    # Handler for time
    def get_time(timezone: str):
        return {"timezone": timezone, "time": "14:30"}

    provider.register_tool(
        name="get_weather",
        description="Get weather",
        input_schema={
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "required": ["location"]
        },
        handler=get_weather
    )

    provider.register_tool(
        name="get_time",
        description="Get current time",
        input_schema={
            "type": "object",
            "properties": {
                "timezone": {"type": "string"}
            },
            "required": ["timezone"]
        },
        handler=get_time
    )

    # Claude may call both tools in parallel
    result = provider.call_with_tools(
        prompt="What's the weather and time in Paris?",
        auto_execute=True
    )

    print(f"Tools called in parallel: {len(result['tool_calls'])}")
    print("\nFinal answer:")
    print(result["final_text"])


def example_command_execution():
    """Example: Command-based workflow execution."""
    print("\n=== Example: Command Execution ===\n")

    provider = AnthropicToolsProvider()

    commands = """
    # Register a tool
    TOOL_REGISTER name="calculator" description="Calculate math" schema_file="calculator_schema.json"

    # Make a call
    TOOL_CALL prompt="What is 25 * 4?" tool_choice="calculator" -> calc_result

    # Print result
    PRINT message="Calculation complete"
    """

    # Note: This would need calculator_schema.json to exist
    # result = provider.execute_commands(commands)
    print("Command execution example (requires schema file)")


def example_load_from_file():
    """Example: Load tools from JSON file."""
    print("\n=== Example: Load Tools from File ===\n")

    # Create example tools file
    tools_data = [
        {
            "name": "search_products",
            "description": "Search product catalog",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "category": {"type": "string"}
                },
                "required": ["query"]
            }
        },
        {
            "name": "check_inventory",
            "description": "Check product inventory",
            "input_schema": {
                "type": "object",
                "properties": {
                    "product_id": {"type": "string"}
                },
                "required": ["product_id"]
            }
        }
    ]

    # Save to file
    with open("example_tools.json", "w") as f:
        json.dump(tools_data, f, indent=2)

    # Load tools
    provider = AnthropicToolsProvider()
    provider.load_tools_from_file("example_tools.json")

    print(f"Loaded {len(provider.tools)} tools:")
    for tool in provider.tools:
        print(f"  - {tool['name']}: {tool['description']}")


def example_force_specific_tool():
    """Example: Force Claude to use a specific tool."""
    print("\n=== Example: Force Specific Tool ===\n")

    provider = AnthropicToolsProvider()

    # Register extraction tool
    tool = create_tool_definition(
        name="extract_contact",
        description="Extract contact information",
        properties={
            "name": {"type": "string"},
            "email": {"type": "string", "format": "email"},
            "phone": {"type": "string"}
        },
        required=["name"]
    )

    provider.register_tool(**tool)

    # Force this specific tool
    result = provider.call_with_tools(
        prompt="Contact: Jane Doe, jane@example.com, 555-1234",
        tool_choice={"type": "tool", "name": "extract_contact"}
    )

    print("Extracted using forced tool:")
    print(json.dumps(result["tool_calls"][0]["input"], indent=2))


def example_error_handling():
    """Example: Handle errors in tool execution."""
    print("\n=== Example: Error Handling ===\n")

    provider = AnthropicToolsProvider()

    # Handler that may fail
    def risky_operation(value: str):
        if not value:
            raise ValueError("Value cannot be empty")
        if value == "error":
            raise RuntimeError("Simulated error")
        return {"processed": value.upper()}

    provider.register_tool(
        name="risky_tool",
        description="A tool that might fail",
        input_schema={
            "type": "object",
            "properties": {
                "value": {"type": "string"}
            },
            "required": ["value"]
        },
        handler=risky_operation
    )

    # This will trigger an error
    result = provider.call_with_tools(
        prompt="Process the value 'error'",
        tool_choice="risky_tool",
        auto_execute=True
    )

    print("Tool results:")
    for tr in result["tool_results"]:
        if tr.get("is_error"):
            print(f"  ERROR: {tr['content']}")
        else:
            print(f"  SUCCESS: {tr['content']}")


def main():
    """Run all examples."""
    examples = [
        example_basic_tool,
        example_auto_execute,
        example_structured_extraction,
        example_multiple_tools,
        example_parallel_tools,
        example_load_from_file,
        example_force_specific_tool,
        example_error_handling,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {e}")
        print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    # Set your API key first
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Please set ANTHROPIC_API_KEY environment variable")
        print("\nExample tools and schemas:")
        print(json.dumps(EXAMPLE_TOOLS, indent=2))
    else:
        main()
