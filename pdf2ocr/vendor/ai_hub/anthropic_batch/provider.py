# Filepath: code_migration/ai_providers/anthropic_batch/provider.py
# Description: AnthropicBatchProvider class for batch processing with Message Batches API
# Layer: AI Provider
# References: reference_codebase/AIMOS/providers/Anthropic/batch/provider.py

"""
Anthropic Batch Processing Provider
Uses the Anthropic Message Batches API for bulk processing with 50% cost savings.
"""

import os
import json
import time
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from anthropic import Anthropic
from dotenv import load_dotenv

from .model_config import get_model_config, get_default_model, BATCH_CONFIG


class AnthropicBatchProvider:
    """Provider for Anthropic batch processing with 50% cost savings."""

    MAX_REQUESTS_PER_BATCH = 100000
    MAX_BATCH_SIZE_MB = 256
    RESULTS_AVAILABLE_HOURS = 24

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the Anthropic Batch Provider.

        Args:
            api_key: Anthropic API key. If not provided, loads from ANTHROPIC_API_KEY env var.
            model: Default model to use. If not provided, uses claude-sonnet-4-5-20250929.

        Raises:
            ValueError: If API key is not provided or found in environment.
        """
        load_dotenv()

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Anthropic API key must be provided or set in ANTHROPIC_API_KEY environment variable"
            )

        self.model = model or os.getenv("ANTHROPIC_MODEL", get_default_model())
        self.client = Anthropic(api_key=self.api_key)

    def create_batch(
        self, requests: List[Dict[str, Any]], model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new batch of requests.

        Supports two request formats:
        1. Simplified: {"custom_id": "...", "prompt": "...", "max_tokens": 1024}
        2. Full format: {"custom_id": "...", "params": {"model": "...", "messages": [...]}}

        Args:
            requests: List of request objects
            model: Default model to use (can be overridden per request)

        Returns:
            Batch object with id, status, and request counts

        Raises:
            ValueError: If requests exceed limits or have invalid format
        """
        used_model = model or self.model

        # Format requests if they're in simplified format
        formatted_requests = []
        for i, req in enumerate(requests):
            if "custom_id" in req and "params" in req:
                # Already formatted - use as-is
                formatted_requests.append(req)
            elif "messages" in req or "prompt" in req:
                # Simple format - convert to full format
                params = {"model": req.get("model", used_model)}

                if "messages" in req:
                    params["messages"] = req["messages"]
                elif "prompt" in req:
                    params["messages"] = [{"role": "user", "content": req["prompt"]}]

                params["max_tokens"] = req.get("max_tokens", 4096)

                if "system" in req:
                    params["system"] = req["system"]
                if "temperature" in req:
                    params["temperature"] = req["temperature"]
                if "top_p" in req:
                    params["top_p"] = req["top_p"]
                if "stop_sequences" in req:
                    params["stop_sequences"] = req["stop_sequences"]

                formatted_requests.append(
                    {"custom_id": req.get("custom_id", f"request-{i}"), "params": params}
                )
            else:
                raise ValueError(
                    f"Invalid request format at index {i}. "
                    "Expected 'prompt' or 'messages' or 'params' field."
                )

        if len(formatted_requests) > self.MAX_REQUESTS_PER_BATCH:
            raise ValueError(
                f"Maximum {self.MAX_REQUESTS_PER_BATCH} requests per batch. "
                f"Got {len(formatted_requests)}."
            )

        # Create batch via API
        batch = self.client.batches.create(requests=formatted_requests)

        return {
            "id": batch.id,
            "status": batch.status,
            "created_at": str(batch.created_at) if hasattr(batch, "created_at") else None,
            "request_counts": {
                "total": (
                    batch.request_counts.total
                    if hasattr(batch.request_counts, "total")
                    else len(formatted_requests)
                ),
                "succeeded": getattr(batch.request_counts, "succeeded", 0),
                "errored": getattr(batch.request_counts, "errored", 0),
                "canceled": getattr(batch.request_counts, "canceled", 0),
                "expired": getattr(batch.request_counts, "expired", 0),
            },
        }

    def create_batch_from_prompts(
        self,
        prompts: List[str],
        model: Optional[str] = None,
        system: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        prefix: str = "prompt",
    ) -> Dict[str, Any]:
        """
        Create a batch from a simple list of prompts.

        Convenience method for quick batch creation from text prompts.

        Args:
            prompts: List of prompt strings
            model: Model to use (defaults to instance default)
            system: System prompt (same for all requests)
            max_tokens: Max tokens per request
            temperature: Sampling temperature (0.0-1.0)
            prefix: Prefix for custom_id (will be "{prefix}-{index}")

        Returns:
            Batch object

        Example:
            batch = provider.create_batch_from_prompts([
                "What is AI?",
                "Explain quantum computing"
            ], system="Be concise", max_tokens=500)
        """
        requests = []
        for i, prompt in enumerate(prompts):
            req = {
                "custom_id": f"{prefix}-{i}",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system:
                req["system"] = system
            if model:
                req["model"] = model
            requests.append(req)

        return self.create_batch(requests)

    def create_batch_from_file(
        self, file_path: str, model: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a batch from a JSON or JSONL file.

        JSON format: Array of request objects or {"requests": [...]}
        JSONL format: One request object per line

        Args:
            file_path: Path to requests file (.json or .jsonl)
            model: Default model to use

        Returns:
            Batch object

        Raises:
            ValueError: If file format is invalid
            FileNotFoundError: If file doesn't exist
        """
        requests = []
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if path.suffix == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        requests.append(json.loads(line))
        else:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    requests = data
                elif "requests" in data:
                    requests = data["requests"]
                else:
                    raise ValueError(
                        "Invalid file format. Expected array or {requests: [...]}"
                    )

        return self.create_batch(requests, model=model)

    def get_status(self, batch_id: str) -> Dict[str, Any]:
        """
        Get the current status of a batch.

        Args:
            batch_id: The batch ID

        Returns:
            Status information with request counts and completion status

        Example:
            status = provider.get_status("batch_abc123")
            if status['is_complete']:
                print(f"Succeeded: {status['request_counts']['succeeded']}")
        """
        batch = self.client.batches.retrieve(batch_id)

        return {
            "id": batch.id,
            "status": batch.status,
            "request_counts": {
                "total": batch.request_counts.total,
                "succeeded": batch.request_counts.succeeded,
                "errored": batch.request_counts.errored,
                "canceled": getattr(batch.request_counts, "canceled", 0),
                "expired": getattr(batch.request_counts, "expired", 0),
            },
            "is_complete": batch.status == "ended",
        }

    def wait_for_completion(
        self,
        batch_id: str,
        poll_interval: int = 60,
        timeout: int = 86400,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Wait for a batch to complete with periodic status checks.

        Polls the batch status at regular intervals until completion or timeout.

        Args:
            batch_id: The batch ID
            poll_interval: Seconds between status checks (default: 60)
            timeout: Maximum wait time in seconds (default: 24 hours)
            verbose: Print progress updates

        Returns:
            Final status when complete

        Raises:
            TimeoutError: If batch doesn't complete within timeout

        Example:
            status = provider.wait_for_completion(
                batch_id,
                poll_interval=30,
                timeout=3600,
                verbose=True
            )
        """
        start_time = time.time()

        while True:
            status = self.get_status(batch_id)

            if verbose:
                counts = status["request_counts"]
                elapsed = int(time.time() - start_time)
                print(
                    f"[{time.strftime('%H:%M:%S')}] [{elapsed}s] "
                    f"Status: {status['status']}, "
                    f"Completed: {counts['succeeded']}/{counts['total']}, "
                    f"Errors: {counts['errored']}"
                )

            if status["is_complete"]:
                return status

            if time.time() - start_time > timeout:
                raise TimeoutError(
                    f"Batch {batch_id} did not complete within {timeout} seconds"
                )

            time.sleep(poll_interval)

    def get_results(
        self, batch_id: str, output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the results of a completed batch.

        Retrieves all results and optionally saves to disk.

        Args:
            batch_id: The batch ID
            output_dir: Optional directory to save results

        Returns:
            List of result objects with custom_id and result

        Example:
            results = provider.get_results(batch_id, output_dir="results/")
            # Results saved to:
            # - results/{batch_id}_results.jsonl (full results)
            # - results/{batch_id}_summary.json (summary stats)
        """
        results_response = self.client.batches.results(batch_id)
        results = []

        for line in results_response.iter_lines():
            if line:
                result = json.loads(line)
                results.append(result)

        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save full results as JSONL
            with open(
                output_path / f"{batch_id}_results.jsonl", "w", encoding="utf-8"
            ) as f:
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

            # Save summary
            succeeded = [
                r for r in results if r.get("result", {}).get("type") == "succeeded"
            ]
            failed = [
                r for r in results if r.get("result", {}).get("type") != "succeeded"
            ]

            with open(
                output_path / f"{batch_id}_summary.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "batch_id": batch_id,
                        "total": len(results),
                        "succeeded": len(succeeded),
                        "failed": len(failed),
                        "success_rate": (
                            len(succeeded) / len(results) * 100 if results else 0
                        ),
                    },
                    f,
                    indent=2,
                )

        return results

    def get_results_text(self, batch_id: str) -> Dict[str, str]:
        """
        Get just the text responses from a batch, keyed by custom_id.

        Convenience method for extracting text content from results.

        Args:
            batch_id: The batch ID

        Returns:
            Dictionary mapping custom_id to response text (or error message)

        Example:
            results = provider.get_results_text(batch_id)
            for custom_id, text in results.items():
                print(f"{custom_id}: {text}")
        """
        results = self.get_results(batch_id)
        text_results = {}

        for result in results:
            custom_id = result.get("custom_id")
            result_data = result.get("result", {})

            if result_data.get("type") == "succeeded":
                message = result_data.get("message", {})
                content = message.get("content", [])
                if content and content[0].get("type") == "text":
                    text_results[custom_id] = content[0].get("text", "")
            else:
                error = result_data.get("error", {})
                text_results[custom_id] = f"ERROR: {error.get('message', 'Unknown error')}"

        return text_results

    def list_batches(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List recent batches.

        Args:
            limit: Maximum number of batches to return (default: 20)

        Returns:
            List of batch info objects with id, status, and counts

        Example:
            batches = provider.list_batches(limit=10)
            for batch in batches:
                print(f"{batch['id']}: {batch['status']}")
        """
        batches = self.client.batches.list(limit=limit)
        return [
            {
                "id": batch.id,
                "status": batch.status,
                "request_counts": {
                    "total": batch.request_counts.total,
                    "succeeded": batch.request_counts.succeeded,
                    "errored": batch.request_counts.errored,
                },
            }
            for batch in batches.data
        ]

    def cancel_batch(self, batch_id: str) -> Dict[str, Any]:
        """
        Cancel a running batch.

        Args:
            batch_id: The batch ID to cancel

        Returns:
            Updated batch status

        Note:
            Only in-progress batches can be canceled. Completed batches cannot be canceled.

        Example:
            status = provider.cancel_batch(batch_id)
            print(f"Batch canceled: {status['status']}")
        """
        self.client.batches.cancel(batch_id)
        return self.get_status(batch_id)

    def execute_commands(
        self, command_string: str, verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a workflow of batch operations using a command DSL.

        Command Format:
        ---------------
        BATCH_CREATE requests_file="requests.json" [output_dir="results/"]
        BATCH_CREATE_PROMPTS prompts_file="prompts.txt" [system="..."] [prefix="item"]
        BATCH_STATUS batch_id="..."
        BATCH_WAIT batch_id="..." [poll_interval=60] [timeout=86400]
        BATCH_RESULTS batch_id="..." [output_dir="results/"]
        BATCH_LIST [limit=20]
        BATCH_CANCEL batch_id="..."

        SET var="value"
        WAIT seconds=N
        PRINT message="..."
        SAVE content="..." file="..."

        Use -> variable_name to store command output
        Use ${variable_name} or ${LAST_OUTPUT} for substitution

        Args:
            command_string: Multi-line string containing commands
            verbose: Print execution progress

        Returns:
            Dictionary with execution results and metadata

        Example:
            workflow = '''
            BATCH_CREATE_PROMPTS prompts_file="prompts.txt" -> batch
            BATCH_WAIT batch_id="${batch.id}"
            BATCH_RESULTS batch_id="${batch.id}" output_dir="results/"
            '''
            result = provider.execute_commands(workflow)
        """
        variables = {}
        results = []
        last_output = None

        def substitute_vars(value: str) -> str:
            """Replace variable placeholders with actual values."""
            if not isinstance(value, str):
                return value
            if last_output and "${LAST_OUTPUT}" in value:
                if isinstance(last_output, dict):
                    value = value.replace("${LAST_OUTPUT}", json.dumps(last_output))
                else:
                    value = value.replace("${LAST_OUTPUT}", str(last_output))
            for var_name, var_value in variables.items():
                if isinstance(var_value, dict):
                    value = value.replace(f"${{{var_name}}}", json.dumps(var_value))
                else:
                    value = value.replace(f"${{{var_name}}}", str(var_value))
            return value

        def parse_args(arg_string: str) -> Dict[str, Any]:
            """Parse command arguments from string."""
            args = {}
            pattern = r'(\w+)=(?:"([^"]*)"|(\S+))'
            matches = re.findall(pattern, arg_string)
            for match in matches:
                key = match[0]
                value = match[1] if match[1] else match[2]
                # Type conversion
                if value.lower() in ("true", "false"):
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif re.match(r"^\d+\.\d+$", value):
                    value = float(value)
                # Variable substitution
                if isinstance(value, str):
                    value = substitute_vars(value)
                args[key] = value
            return args

        def parse_arrow_assignment(line: str):
            """Parse -> variable assignment from line."""
            arrow_match = re.search(r"->\s*(\w+)\s*$", line)
            if arrow_match:
                var_name = arrow_match.group(1)
                line = line[: arrow_match.start()].strip()
                return line, var_name
            return line, None

        lines = command_string.strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
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

                # Control flow commands
                if command == "SET":
                    for key, value in args.items():
                        variables[key] = value
                        if verbose:
                            print(f"  Set ${{{key}}} = {value}")

                elif command == "WAIT":
                    seconds = args.get("seconds", 1)
                    if verbose:
                        print(f"  Waiting {seconds} seconds...")
                    time.sleep(seconds)

                elif command == "PRINT":
                    message = args.get("message", args.get("text", ""))
                    print(f"  {message}")

                elif command == "SAVE":
                    content = args.get("content", "")
                    file_path = args.get("file", "")
                    if file_path:
                        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
                        if isinstance(content, (dict, list)):
                            content = json.dumps(content, indent=2)
                        with open(file_path, "w", encoding="utf-8") as f:
                            f.write(str(content))
                        if verbose:
                            print(f"  Saved to: {file_path}")

                # Batch commands
                elif command == "BATCH_CREATE":
                    requests_file = args.get("requests_file", "")
                    batch = self.create_batch_from_file(requests_file)

                    if verbose:
                        print(f"  [OK] Created batch: {batch['id']}")
                        print(f"  Requests: {batch['request_counts']['total']}")

                    if assign_var:
                        variables[assign_var] = batch
                    last_output = batch

                    results.append(
                        {"command": "BATCH_CREATE", "line": line_num, "result": batch}
                    )

                elif command == "BATCH_CREATE_PROMPTS":
                    prompts_file = args.get("prompts_file", "")
                    system = args.get("system", None)
                    prefix = args.get("prefix", "prompt")

                    # Read prompts from file (one per line)
                    with open(prompts_file, "r", encoding="utf-8") as f:
                        prompts = [line.strip() for line in f if line.strip()]

                    batch = self.create_batch_from_prompts(
                        prompts=prompts, system=system, prefix=prefix
                    )

                    if verbose:
                        print(
                            f"  [OK] Created batch from {len(prompts)} prompts: {batch['id']}"
                        )

                    if assign_var:
                        variables[assign_var] = batch
                    last_output = batch

                    results.append(
                        {
                            "command": "BATCH_CREATE_PROMPTS",
                            "line": line_num,
                            "result": batch,
                        }
                    )

                elif command == "BATCH_STATUS":
                    batch_id = args.get("batch_id", "")
                    status = self.get_status(batch_id)

                    if verbose:
                        counts = status["request_counts"]
                        print(f"  Status: {status['status']}")
                        print(f"  Completed: {counts['succeeded']}/{counts['total']}")
                        if counts["errored"] > 0:
                            print(f"  Errors: {counts['errored']}")

                    if assign_var:
                        variables[assign_var] = status
                    last_output = status

                    results.append(
                        {"command": "BATCH_STATUS", "line": line_num, "result": status}
                    )

                elif command == "BATCH_WAIT":
                    batch_id = args.get("batch_id", "")
                    poll_interval = args.get("poll_interval", 60)
                    timeout = args.get("timeout", 86400)

                    if verbose:
                        print(f"  Waiting for batch {batch_id} to complete...")

                    status = self.wait_for_completion(
                        batch_id,
                        poll_interval=poll_interval,
                        timeout=timeout,
                        verbose=verbose,
                    )

                    if verbose:
                        print(f"  [OK] Batch completed!")

                    if assign_var:
                        variables[assign_var] = status
                    last_output = status

                    results.append(
                        {"command": "BATCH_WAIT", "line": line_num, "result": status}
                    )

                elif command == "BATCH_RESULTS":
                    batch_id = args.get("batch_id", "")
                    output_dir = args.get("output_dir", None)

                    batch_results = self.get_results(batch_id, output_dir=output_dir)

                    succeeded = len(
                        [
                            r
                            for r in batch_results
                            if r.get("result", {}).get("type") == "succeeded"
                        ]
                    )
                    failed = len(batch_results) - succeeded

                    if verbose:
                        print(f"  [OK] Retrieved {len(batch_results)} results")
                        print(f"  Succeeded: {succeeded}, Failed: {failed}")
                        if output_dir:
                            print(f"  Saved to: {output_dir}")

                    if assign_var:
                        variables[assign_var] = batch_results
                    last_output = batch_results

                    results.append(
                        {
                            "command": "BATCH_RESULTS",
                            "line": line_num,
                            "result": {
                                "total": len(batch_results),
                                "succeeded": succeeded,
                                "failed": failed,
                            },
                        }
                    )

                elif command == "BATCH_LIST":
                    limit = args.get("limit", 20)
                    batches = self.list_batches(limit=limit)

                    if verbose:
                        print(f"  Found {len(batches)} batches:")
                        for batch in batches:
                            print(f"    {batch['id']}: {batch['status']}")

                    if assign_var:
                        variables[assign_var] = batches
                    last_output = batches

                    results.append(
                        {
                            "command": "BATCH_LIST",
                            "line": line_num,
                            "result": {"batches": batches},
                        }
                    )

                elif command == "BATCH_CANCEL":
                    batch_id = args.get("batch_id", "")
                    status = self.cancel_batch(batch_id)

                    if verbose:
                        print(f"  [OK] Canceled batch {batch_id}")

                    results.append(
                        {"command": "BATCH_CANCEL", "line": line_num, "result": status}
                    )

                else:
                    if verbose:
                        print(f"  [WARN] Unknown command: {command}")

            except Exception as e:
                error_msg = f"Error on line {line_num} ({command}): {str(e)}"
                if verbose:
                    print(f"  [ERROR] {error_msg}")
                results.append({"command": command, "line": line_num, "error": str(e)})

        return {
            "success": True,
            "results": results,
            "variables": variables,
            "last_output": last_output,
            "total_commands": len([r for r in results if "error" not in r]),
            "total_errors": len([r for r in results if "error" in r]),
        }


def main():
    """Example usage of the Anthropic Batch Provider."""
    try:
        provider = AnthropicBatchProvider()

        print("=== Anthropic Batch Provider ===")
        print(f"Default model: {provider.model}")
        print()

        print("Key Benefits:")
        print("  - 50% cost savings compared to standard API")
        print("  - Up to 100,000 requests per batch")
        print("  - Most batches complete within 1 hour")
        print()

        print("Limits:")
        print(f"  Max requests per batch: {provider.MAX_REQUESTS_PER_BATCH:,}")
        print(f"  Max batch size: {provider.MAX_BATCH_SIZE_MB} MB")
        print(f"  Results available: {provider.RESULTS_AVAILABLE_HOURS} hours")
        print()

        print("Example usage:")
        print("  provider = AnthropicBatchProvider()")
        print("  batch = provider.create_batch_from_prompts(['What is 2+2?'])")
        print("  status = provider.wait_for_completion(batch['id'])")
        print("  results = provider.get_results_text(batch['id'])")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
