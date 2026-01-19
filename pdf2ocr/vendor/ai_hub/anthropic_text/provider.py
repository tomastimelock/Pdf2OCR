"""
Anthropic Text Generation Provider
Uses the Anthropic Messages API for text generation with Claude models.
"""

import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from anthropic import Anthropic
from dotenv import load_dotenv

# Use relative imports for self-contained module
from .model_config import (
    get_all_models, get_model_config, get_default_model,
    get_help_text, get_param_options, get_model_context_window,
    get_model_max_output
)


class AnthropicTextProvider:
    """Provider for Anthropic text generation using the Messages API."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Anthropic Text Provider.

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

    @staticmethod
    def list_models() -> Dict[str, Any]:
        """List all available text generation models with their configurations."""
        all_models = get_all_models()
        return {name: {
            "description": config.description,
            "context_window": config.context_window,
            "max_output": config.max_output,
            "supported_params": config.supported_params,
            "default_params": config.default_params,
            "notes": config.notes
        } for name, config in all_models.items()}

    @staticmethod
    def get_help() -> str:
        """Get help text for text generation commands."""
        return get_help_text("text")

    @staticmethod
    def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a specific model."""
        config = get_model_config(model_name)
        if config:
            return {
                "name": config.name,
                "category": config.category,
                "description": config.description,
                "context_window": config.context_window,
                "max_output": config.max_output,
                "supported_params": config.supported_params,
                "default_params": config.default_params,
                "param_options": config.param_options,
                "notes": config.notes,
                "knowledge_cutoff": config.knowledge_cutoff
            }
        return None

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop_sequences: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """
        Generate text using the Anthropic Messages API.

        Args:
            prompt: The input prompt (user message)
            model: Model to use (defaults to instance model)
            system: System prompt for context and instructions
            max_tokens: Maximum tokens to generate (default: 4096)
            temperature: Randomness 0.0-1.0 (mutually exclusive with top_p)
            top_p: Nucleus sampling (mutually exclusive with temperature)
            top_k: Top-k sampling
            stop_sequences: Sequences that stop generation
            **kwargs: Additional parameters to pass to the API

        Returns:
            The generated text output
        """
        used_model = model or self.model

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system:
            params["system"] = system

        # Temperature and top_p are mutually exclusive
        if temperature is not None:
            params["temperature"] = temperature
        elif top_p is not None:
            params["top_p"] = top_p

        if top_k is not None:
            params["top_k"] = top_k

        if stop_sequences:
            params["stop_sequences"] = stop_sequences

        # Add any extra kwargs
        params.update(kwargs)

        response = self.client.messages.create(**params)

        # Extract text from response content
        return self._extract_text(response)

    def generate_with_messages(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate text using structured messages for multi-turn conversations.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Roles can be: 'user', 'assistant'
            model: Model to use (defaults to instance model)
            system: System prompt for context and instructions
            max_tokens: Maximum tokens to generate
            temperature: Randomness 0.0-1.0
            **kwargs: Additional parameters to pass to the API

        Returns:
            The generated text output
        """
        used_model = model or self.model

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "messages": messages
        }

        if system:
            params["system"] = system

        if temperature is not None:
            params["temperature"] = temperature

        params.update(kwargs)

        response = self.client.messages.create(**params)
        return self._extract_text(response)

    def generate_with_prefill(
        self,
        prompt: str,
        prefill: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        """
        Generate text with response prefilling to guide output format.

        Args:
            prompt: The user's input prompt
            prefill: The start of the assistant's response (cannot end with whitespace)
            model: Model to use (defaults to instance model)
            system: System prompt for context
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            The generated text output (including prefill)
        """
        # Ensure prefill doesn't end with whitespace
        prefill = prefill.rstrip()

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": prefill}
        ]

        used_model = model or self.model

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "messages": messages
        }

        if system:
            params["system"] = system

        params.update(kwargs)

        response = self.client.messages.create(**params)

        # Prepend the prefill to the response
        return prefill + self._extract_text(response)

    def stream_generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs
    ):
        """
        Stream text generation for real-time output.

        Args:
            prompt: The input prompt
            model: Model to use
            system: System prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Yields:
            Text chunks as they are generated
        """
        used_model = model or self.model

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system:
            params["system"] = system

        params.update(kwargs)

        with self.client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                yield text

    def get_full_response(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        **kwargs
    ) -> Any:
        """
        Get the full response object from the API (not just text).

        Args:
            prompt: The input prompt
            model: Model to use
            system: System prompt
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            The full response object from Anthropic
        """
        used_model = model or self.model

        params = {
            "model": used_model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}]
        }

        if system:
            params["system"] = system

        params.update(kwargs)

        return self.client.messages.create(**params)

    def _extract_text(self, response) -> str:
        """Extract text content from a response object."""
        text_parts = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_parts.append(block.text)
        return ''.join(text_parts)

    def execute_commands(self, command_string: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Execute a workflow of text operations from a command string.

        Command Format:
        ---------------
        TEXT_GENERATE prompt="..." [output="file.txt"] [model="..."] [system="..."] [max_tokens=N]
        TEXT_GENERATE prompt="..." -> variable_name
        TEXT_MESSAGES messages="..." [output="file.txt"] [system="..."]
        TEXT_STREAM prompt="..." [system="..."]

        SET var="value"
        WAIT seconds=N
        PRINT message="..."
        SAVE content="..." file="..."

        Args:
            command_string: Multi-line string containing commands
            verbose: Print execution progress (default: True)

        Returns:
            Dictionary with execution results and metadata
        """
        # Variables storage
        variables = {}
        results = []
        last_output = None

        # Helper function to substitute variables
        def substitute_vars(value: str) -> str:
            if not isinstance(value, str):
                return value

            if last_output and "${LAST_OUTPUT}" in value:
                value = value.replace("${LAST_OUTPUT}", str(last_output))

            for var_name, var_value in variables.items():
                value = value.replace(f"${{{var_name}}}", str(var_value))

            return value

        # Helper function to parse arguments
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

        # Check for arrow syntax (-> variable_name)
        def parse_arrow_assignment(line: str):
            arrow_match = re.search(r'->\s*(\w+)\s*$', line)
            if arrow_match:
                var_name = arrow_match.group(1)
                line = line[:arrow_match.start()].strip()
                return line, var_name
            return line, None

        # Process each command line
        lines = command_string.strip().split('\n')

        for line_num, line in enumerate(lines, 1):
            line = line.strip()

            if not line or line.startswith('#') or line.startswith('//'):
                continue

            try:
                # Check for arrow assignment
                line, assign_var = parse_arrow_assignment(line)

                parts = line.split(None, 1)
                if not parts:
                    continue

                command = parts[0].upper()
                arg_string = parts[1] if len(parts) > 1 else ""
                args = parse_args(arg_string)

                if verbose:
                    print(f"[Line {line_num}] Executing: {command}")

                # Execute commands
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
                        # Ensure directory exists
                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        if verbose:
                            print(f"  Saved to: {file_path}")

                elif command == "TEXT_HELP":
                    help_text = self.get_help()
                    if verbose:
                        print(help_text)
                    results.append({
                        'command': 'TEXT_HELP',
                        'line': line_num,
                        'result': {'text': help_text}
                    })

                elif command == "TEXT_MODELS":
                    models = self.list_models()
                    if verbose:
                        print("\nAvailable Claude Models:")
                        print("-" * 50)
                        for name, info in models.items():
                            print(f"  {name}")
                            print(f"    {info['description']}")
                            print(f"    Context: {info['context_window']:,} tokens, Max output: {info['max_output']:,}")
                            if info.get('notes'):
                                print(f"    Note: {info['notes']}")
                            print()
                    results.append({
                        'command': 'TEXT_MODELS',
                        'line': line_num,
                        'result': {'models': models}
                    })

                elif command == "TEXT_GENERATE":
                    prompt = args.get('prompt', args.get('input', ''))
                    output = args.get('output', None)

                    gen_params = {'prompt': prompt}

                    if 'model' in args:
                        gen_params['model'] = args['model']
                    if 'system' in args:
                        gen_params['system'] = args['system']
                    if 'max_tokens' in args:
                        gen_params['max_tokens'] = args['max_tokens']
                    if 'temperature' in args:
                        gen_params['temperature'] = args['temperature']
                    if 'top_p' in args:
                        gen_params['top_p'] = args['top_p']

                    response = self.generate(**gen_params)

                    # Handle arrow assignment
                    if assign_var:
                        variables[assign_var] = response
                        last_output = response
                        if verbose:
                            preview = response[:100] + "..." if len(response) > 100 else response
                            print(f"  [OK] Stored in ${{{assign_var}}}: {preview}")
                    elif output:
                        # Ensure directory exists
                        Path(output).parent.mkdir(parents=True, exist_ok=True)
                        with open(output, 'w', encoding='utf-8') as f:
                            f.write(response)
                        last_output = output
                        if verbose:
                            print(f"  [OK] Generated text saved to: {output}")
                    else:
                        last_output = response
                        if verbose:
                            preview = response[:100] + "..." if len(response) > 100 else response
                            print(f"  [OK] Generated: {preview}")

                    results.append({
                        'command': 'TEXT_GENERATE',
                        'line': line_num,
                        'result': {'output': output or assign_var, 'text': response}
                    })

                elif command == "TEXT_MESSAGES":
                    # Parse messages from a JSON-like string or from args
                    messages = []
                    messages_str = args.get('messages', '')

                    if messages_str:
                        import json
                        try:
                            messages = json.loads(messages_str)
                        except json.JSONDecodeError:
                            # Try parsing as simple format: "user:hello,assistant:hi"
                            for part in messages_str.split(','):
                                if ':' in part:
                                    role, content = part.split(':', 1)
                                    messages.append({'role': role.strip(), 'content': content.strip()})

                    # Also support individual role args
                    if 'user' in args:
                        messages.append({'role': 'user', 'content': args['user']})

                    output = args.get('output', None)

                    gen_params = {'messages': messages}
                    if 'model' in args:
                        gen_params['model'] = args['model']
                    if 'system' in args:
                        gen_params['system'] = args['system']
                    if 'max_tokens' in args:
                        gen_params['max_tokens'] = args['max_tokens']

                    response = self.generate_with_messages(**gen_params)

                    if assign_var:
                        variables[assign_var] = response
                        last_output = response
                    elif output:
                        Path(output).parent.mkdir(parents=True, exist_ok=True)
                        with open(output, 'w', encoding='utf-8') as f:
                            f.write(response)
                        last_output = output
                        if verbose:
                            print(f"  [OK] Generated text saved to: {output}")
                    else:
                        last_output = response
                        if verbose:
                            preview = response[:100] + "..." if len(response) > 100 else response
                            print(f"  [OK] Generated: {preview}")

                    results.append({
                        'command': 'TEXT_MESSAGES',
                        'line': line_num,
                        'result': {'output': output, 'text': response}
                    })

                elif command == "TEXT_STREAM":
                    prompt = args.get('prompt', '')
                    system = args.get('system', None)

                    if verbose:
                        print("  Streaming response:")
                        print("  " + "-" * 40)

                    full_response = ""
                    for chunk in self.stream_generate(prompt=prompt, system=system):
                        full_response += chunk
                        if verbose:
                            print(chunk, end="", flush=True)

                    if verbose:
                        print("\n  " + "-" * 40)

                    if assign_var:
                        variables[assign_var] = full_response
                    last_output = full_response

                    results.append({
                        'command': 'TEXT_STREAM',
                        'line': line_num,
                        'result': {'text': full_response}
                    })

                elif command == "TEXT_PREFILL":
                    prompt = args.get('prompt', '')
                    prefill = args.get('prefill', '')
                    output = args.get('output', None)

                    gen_params = {'prompt': prompt, 'prefill': prefill}
                    if 'model' in args:
                        gen_params['model'] = args['model']
                    if 'system' in args:
                        gen_params['system'] = args['system']

                    response = self.generate_with_prefill(**gen_params)

                    if assign_var:
                        variables[assign_var] = response
                        last_output = response
                    elif output:
                        Path(output).parent.mkdir(parents=True, exist_ok=True)
                        with open(output, 'w', encoding='utf-8') as f:
                            f.write(response)
                        last_output = output
                        if verbose:
                            print(f"  [OK] Generated text saved to: {output}")
                    else:
                        last_output = response
                        if verbose:
                            preview = response[:100] + "..." if len(response) > 100 else response
                            print(f"  [OK] Generated: {preview}")

                    results.append({
                        'command': 'TEXT_PREFILL',
                        'line': line_num,
                        'result': {'output': output, 'text': response}
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


def main():
    """Example usage of the Anthropic Text Provider."""
    try:
        # Initialize the provider
        provider = AnthropicTextProvider()

        print("=== Example 1: Simple text generation ===")
        response = provider.generate(
            prompt="Write a one-sentence story about a robot learning to paint."
        )
        print(f"Response: {response}\n")

        print("=== Example 2: Generation with system prompt ===")
        response = provider.generate(
            prompt="What's the best way to learn programming?",
            system="You are a helpful coding mentor. Give concise, practical advice.",
            max_tokens=200
        )
        print(f"Response: {response}\n")

        print("=== Example 3: Multi-turn conversation ===")
        response = provider.generate_with_messages(
            messages=[
                {"role": "user", "content": "What is 2 + 2?"},
                {"role": "assistant", "content": "2 + 2 equals 4."},
                {"role": "user", "content": "And if I multiply that by 3?"}
            ]
        )
        print(f"Response: {response}\n")

        print("=== Example 4: Response prefilling ===")
        response = provider.generate_with_prefill(
            prompt="List 3 programming languages as JSON",
            prefill='{"languages": ['
        )
        print(f"Response: {response}\n")

        print("=== Example 5: Get full response object ===")
        full_response = provider.get_full_response(
            prompt="Say hello in French."
        )
        print(f"Stop reason: {full_response.stop_reason}")
        print(f"Usage: {full_response.usage}")
        print(f"Text: {full_response.content[0].text}\n")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
