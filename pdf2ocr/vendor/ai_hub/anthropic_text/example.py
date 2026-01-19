"""
Example usage of the Anthropic Text Provider module.
Demonstrates various features and use cases.
"""

from anthropic_text import AnthropicTextProvider, get_default_model, get_fastest_model


def example_basic_generation():
    """Example 1: Simple text generation."""
    print("=" * 60)
    print("Example 1: Basic Text Generation")
    print("=" * 60)

    provider = AnthropicTextProvider()

    response = provider.generate(
        prompt="Write a one-sentence story about a robot learning to paint."
    )
    print(f"Response: {response}\n")


def example_system_prompt():
    """Example 2: Using system prompts for context."""
    print("=" * 60)
    print("Example 2: System Prompt")
    print("=" * 60)

    provider = AnthropicTextProvider()

    response = provider.generate(
        prompt="What's the best way to learn programming?",
        system="You are a helpful coding mentor. Give concise, practical advice.",
        max_tokens=200,
        temperature=0.7
    )
    print(f"Response: {response}\n")


def example_multi_turn():
    """Example 3: Multi-turn conversation."""
    print("=" * 60)
    print("Example 3: Multi-turn Conversation")
    print("=" * 60)

    provider = AnthropicTextProvider()

    response = provider.generate_with_messages([
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
        {"role": "user", "content": "And if I multiply that by 3?"}
    ])
    print(f"Response: {response}\n")


def example_prefilling():
    """Example 4: Response prefilling for structured output."""
    print("=" * 60)
    print("Example 4: Response Prefilling")
    print("=" * 60)

    provider = AnthropicTextProvider()

    response = provider.generate_with_prefill(
        prompt="List 3 programming languages as JSON",
        prefill='{"languages": ['
    )
    print(f"Response: {response}\n")


def example_streaming():
    """Example 5: Streaming responses."""
    print("=" * 60)
    print("Example 5: Streaming")
    print("=" * 60)

    provider = AnthropicTextProvider()

    print("Streaming response: ", end="")
    for chunk in provider.stream_generate(
        prompt="Tell me a very short joke about programmers."
    ):
        print(chunk, end="", flush=True)
    print("\n")


def example_full_response():
    """Example 6: Getting full response metadata."""
    print("=" * 60)
    print("Example 6: Full Response Object")
    print("=" * 60)

    provider = AnthropicTextProvider()

    full_response = provider.get_full_response(
        prompt="Say hello in French."
    )

    print(f"Model: {full_response.model}")
    print(f"Stop reason: {full_response.stop_reason}")
    print(f"Input tokens: {full_response.usage.input_tokens}")
    print(f"Output tokens: {full_response.usage.output_tokens}")
    print(f"Text: {full_response.content[0].text}\n")


def example_different_models():
    """Example 7: Using different models."""
    print("=" * 60)
    print("Example 7: Different Models")
    print("=" * 60)

    provider = AnthropicTextProvider()

    prompt = "What is the capital of France?"

    # Default model (Sonnet 4.5)
    response1 = provider.generate(prompt, model=get_default_model())
    print(f"Sonnet 4.5: {response1}")

    # Fast model (Haiku 4.5)
    response2 = provider.generate(prompt, model=get_fastest_model())
    print(f"Haiku 4.5: {response2}\n")


def example_temperature_control():
    """Example 8: Temperature for creativity control."""
    print("=" * 60)
    print("Example 8: Temperature Control")
    print("=" * 60)

    provider = AnthropicTextProvider()

    prompt = "Describe a sunset in one sentence."

    # Low temperature - more deterministic
    response1 = provider.generate(prompt, temperature=0.0)
    print(f"Temperature 0.0: {response1}")

    # High temperature - more creative
    response2 = provider.generate(prompt, temperature=1.0)
    print(f"Temperature 1.0: {response2}\n")


def example_command_execution():
    """Example 9: Command execution DSL."""
    print("=" * 60)
    print("Example 9: Command Execution")
    print("=" * 60)

    provider = AnthropicTextProvider()

    commands = """
    # Generate a topic
    TEXT_GENERATE prompt="Suggest one interesting science topic (just the topic name)" -> topic

    # Explain the topic
    TEXT_GENERATE prompt="Explain ${topic} in 2 sentences" system="You are a science educator"
    """

    result = provider.execute_commands(commands, verbose=True)

    print(f"\nExecution summary:")
    print(f"Total commands: {result['total_commands']}")
    print(f"Total errors: {result['total_errors']}")
    print(f"Variables: {result['variables']}")


def example_list_models():
    """Example 10: Listing available models."""
    print("=" * 60)
    print("Example 10: List Available Models")
    print("=" * 60)

    models = AnthropicTextProvider.list_models()

    print("Available Claude Models:")
    print("-" * 60)
    for name, info in models.items():
        if not info['description'].startswith('[DEPRECATED]'):
            print(f"\n{name}")
            print(f"  {info['description']}")
            print(f"  Context: {info['context_window']:,} tokens")
            print(f"  Max output: {info['max_output']:,} tokens")


def main():
    """Run all examples."""
    try:
        example_basic_generation()
        example_system_prompt()
        example_multi_turn()
        example_prefilling()
        example_streaming()
        example_full_response()
        example_different_models()
        example_temperature_control()
        example_command_execution()
        example_list_models()

        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\nError running examples: {e}")
        print("\nMake sure you have set ANTHROPIC_API_KEY in your .env file or environment.")


if __name__ == "__main__":
    main()
