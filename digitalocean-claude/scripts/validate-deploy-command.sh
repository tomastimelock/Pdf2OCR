#!/bin/bash
# validate-deploy-command.sh
#
# PreToolUse hook for the do-deploy agent.
# Validates deployment commands for safety.
#
# Exit codes:
#   0 - Allow the command
#   2 - Block the command

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
    exit 0
fi

# Block destructive git operations during deployment
if echo "$COMMAND" | grep -iE 'git.*(push.*--force|reset.*--hard|clean.*-f)' > /dev/null 2>&1; then
    echo "Blocked: Destructive git operations are not allowed during deployment" >&2
    exit 2
fi

# Block direct database modifications
if echo "$COMMAND" | grep -iE '(DROP|TRUNCATE|DELETE FROM)' > /dev/null 2>&1; then
    echo "Blocked: Direct database modifications are not allowed during deployment" >&2
    exit 2
fi

# Block system-level commands
if echo "$COMMAND" | grep -iE '(rm -rf /|shutdown|reboot|halt)' > /dev/null 2>&1; then
    echo "Blocked: System-level destructive commands are not allowed" >&2
    exit 2
fi

exit 0
