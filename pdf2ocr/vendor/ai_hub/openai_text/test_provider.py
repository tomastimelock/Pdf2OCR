# Filepath: code_migration/ai_providers/openai_text/test_provider.py
# Description: Simple test script for OpenAI Text Provider
# Layer: AI Provider - Testing

"""
Test Script for OpenAI Text Provider
=====================================

Simple tests to verify the provider works correctly.
Run with: python test_provider.py
"""

import sys
from typing import List, Dict

try:
    from provider import OpenAITextProvider
    from model_config import (
        get_model_config,
        get_default_model,
        list_all_models,
        get_recommended_model,
        get_preset
    )
except ImportError:
    print("Error: Could not import provider modules.")
    print("Make sure you're running from the openai_text directory.")
    sys.exit(1)


def test_initialization():
    """Test provider initialization."""
    print("\n" + "=" * 70)
    print("Test 1: Initialization")
    print("=" * 70)

    try:
        provider = OpenAITextProvider()
        print(f"✓ Provider initialized successfully")
        print(f"  Default model: {provider.default_model}")
        return True
    except ValueError as e:
        print(f"✗ Initialization failed: {e}")
        print("\n  Please set OPENAI_API_KEY environment variable:")
        print("    export OPENAI_API_KEY='your-api-key-here'")
        print("  Or create a .env file with:")
        print("    OPENAI_API_KEY=your-api-key-here")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def test_model_config():
    """Test model configuration functions."""
    print("\n" + "=" * 70)
    print("Test 2: Model Configuration")
    print("=" * 70)

    try:
        # Test default model
        default = get_default_model()
        print(f"✓ Default model: {default}")

        # Test model config
        config = get_model_config("gpt-4o")
        if config:
            print(f"✓ Model config for gpt-4o:")
            print(f"    Description: {config.description}")
            print(f"    Max tokens: {config.max_tokens}")
            print(f"    Context window: {config.context_window}")
        else:
            print("✗ Could not get model config")
            return False

        # Test list all models
        models = list_all_models()
        print(f"✓ Found {len(models)} available models:")
        for name in models.keys():
            print(f"    - {name}")

        # Test model recommendations
        recommendations = {
            "extraction": get_recommended_model("extraction"),
            "classification": get_recommended_model("classification"),
            "generation": get_recommended_model("generation"),
        }
        print(f"✓ Model recommendations:")
        for task, model in recommendations.items():
            print(f"    {task}: {model}")

        # Test presets
        extraction_preset = get_preset("extraction")
        print(f"✓ Extraction preset: {extraction_preset}")

        return True

    except Exception as e:
        print(f"✗ Model config test failed: {e}")
        return False


def test_basic_completion():
    """Test basic text completion."""
    print("\n" + "=" * 70)
    print("Test 3: Basic Completion")
    print("=" * 70)

    try:
        provider = OpenAITextProvider()
        prompt = "Say exactly: 'Test successful'"

        print(f"Prompt: {prompt}")
        response = provider.complete(prompt, temperature=0.0, max_tokens=50)

        print(f"✓ Response received: {response}")

        if len(response) > 0:
            return True
        else:
            print("✗ Empty response")
            return False

    except Exception as e:
        print(f"✗ Completion test failed: {e}")
        return False


def test_chat_with_system():
    """Test chat with system message."""
    print("\n" + "=" * 70)
    print("Test 4: Chat with System Message")
    print("=" * 70)

    try:
        provider = OpenAITextProvider()

        system_prompt = "You are a helpful assistant. Be concise."
        user_message = "What is 2+2? Answer with just the number."

        print(f"System: {system_prompt}")
        print(f"User: {user_message}")

        response = provider.chat_with_system(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.0,
            max_tokens=20
        )

        print(f"✓ Response: {response}")
        return True

    except Exception as e:
        print(f"✗ Chat with system test failed: {e}")
        return False


def test_swedish_support():
    """Test Swedish language support."""
    print("\n" + "=" * 70)
    print("Test 5: Swedish Language Support")
    print("=" * 70)

    try:
        provider = OpenAITextProvider()

        system_prompt = "Du är en svensk assistent."
        user_message = "Säg 'Hej!' på svenska."

        print(f"System: {system_prompt}")
        print(f"User: {user_message}")

        response = provider.chat_with_system(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.3,
            max_tokens=50
        )

        print(f"✓ Response: {response}")

        # Check if response contains Swedish characters
        swedish_chars = any(char in response for char in ['å', 'ä', 'ö', 'Å', 'Ä', 'Ö'])
        if swedish_chars or 'Hej' in response:
            print("✓ Swedish characters detected")
        else:
            print("  Note: No Swedish characters in response (may be okay)")

        return True

    except Exception as e:
        print(f"✗ Swedish support test failed: {e}")
        return False


def test_multi_turn_chat():
    """Test multi-turn conversation."""
    print("\n" + "=" * 70)
    print("Test 6: Multi-turn Conversation")
    print("=" * 70)

    try:
        provider = OpenAITextProvider()

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
            {"role": "user", "content": "What is my name?"}
        ]

        print("Testing conversation memory...")
        response = provider.chat(messages=messages, temperature=0.0, max_tokens=50)

        print(f"✓ Response: {response}")

        if "Alice" in response or "alice" in response:
            print("✓ Conversation memory working")
        else:
            print("  Note: May not have detected name (response valid)")

        return True

    except Exception as e:
        print(f"✗ Multi-turn chat test failed: {e}")
        return False


def test_model_info():
    """Test getting model information."""
    print("\n" + "=" * 70)
    print("Test 7: Model Information")
    print("=" * 70)

    try:
        provider = OpenAITextProvider()

        # Get all models
        models = provider.get_models()
        print(f"✓ Retrieved {len(models)} models")

        # Get specific model info
        info = provider.get_model_info("gpt-4o")
        if info:
            print(f"✓ GPT-4o info:")
            print(f"    Max tokens: {info['max_tokens']}")
            print(f"    Context: {info['context_window']}")
            print(f"    Cost (input): ${info['cost_per_1k_input']}/1K")
        else:
            print("✗ Could not get model info")
            return False

        # Test cost estimation
        test_messages = [
            {"role": "user", "content": "Hello " * 100}
        ]
        estimate = provider.estimate_cost(test_messages, max_tokens=500)
        print(f"✓ Cost estimate for test query:")
        print(f"    Estimated cost: ${estimate['estimated_cost_usd']:.4f}")

        return True

    except Exception as e:
        print(f"✗ Model info test failed: {e}")
        return False


def test_full_response():
    """Test getting full response object."""
    print("\n" + "=" * 70)
    print("Test 8: Full Response Object")
    print("=" * 70)

    try:
        provider = OpenAITextProvider()

        messages = [{"role": "user", "content": "Say 'test'"}]
        response = provider.get_full_response(
            messages=messages,
            temperature=0.0,
            max_tokens=20
        )

        print(f"✓ Full response received")
        print(f"    Model: {response.model}")
        print(f"    Total tokens: {response.usage.total_tokens}")
        print(f"    Finish reason: {response.choices[0].finish_reason}")
        print(f"    Content: {response.choices[0].message.content}")

        return True

    except Exception as e:
        print(f"✗ Full response test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "=" * 70)
    print("OpenAI Text Provider - Test Suite")
    print("=" * 70)

    tests = [
        ("Initialization", test_initialization),
        ("Model Configuration", test_model_config),
        ("Basic Completion", test_basic_completion),
        ("Chat with System", test_chat_with_system),
        ("Swedish Support", test_swedish_support),
        ("Multi-turn Chat", test_multi_turn_chat),
        ("Model Information", test_model_info),
        ("Full Response", test_full_response),
    ]

    results = []

    # Run initialization first
    init_result = test_initialization()
    if not init_result:
        print("\n" + "=" * 70)
        print("Initialization failed. Cannot run API tests.")
        print("Please configure OPENAI_API_KEY and try again.")
        print("=" * 70)
        return

    # Run remaining tests
    for test_name, test_func in tests[1:]:  # Skip initialization test
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nUnexpected error in {test_name}: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")

    print("\nDetailed results:")
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {test_name}")

    if passed == total:
        print("\n" + "=" * 70)
        print("All tests passed!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print(f"{total - passed} test(s) failed.")
        print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
