#!/bin/bash
# validate-do-command.sh
#
# PreToolUse hook script for validating DigitalOcean commands.
# Blocks dangerous operations unless they are explicitly allowed.
#
# Exit codes:
#   0 - Allow the command
#   2 - Block the command (message sent to Claude via stderr)

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

if [ -z "$COMMAND" ]; then
    exit 0
fi

# Block direct API token exposure
if echo "$COMMAND" | grep -iE '(DIGITALOCEAN_API_TOKEN|DO_API_TOKEN)' | grep -iE '(echo|print|cat|curl.*Bearer)' > /dev/null 2>&1; then
    echo "Blocked: Cannot expose API tokens in commands" >&2
    exit 2
fi

# Block mass deletion commands
if echo "$COMMAND" | grep -iE 'doctl.*delete.*--force' | grep -iE '\-\-tag-name' > /dev/null 2>&1; then
    echo "Blocked: Mass deletion by tag requires manual confirmation. Use the DigitalOcean control panel." >&2
    exit 2
fi

# Block auth modifications
if echo "$COMMAND" | grep -iE 'doctl auth (init|switch|revoke)' > /dev/null 2>&1; then
    echo "Blocked: Authentication changes are not allowed through Claude Code" >&2
    exit 2
fi

# Warn about production resource deletion (let through but log)
if echo "$COMMAND" | grep -iE 'doctl.*(delete|destroy)' > /dev/null 2>&1; then
    if echo "$COMMAND" | grep -iE '(prod|production)' > /dev/null 2>&1; then
        echo "WARNING: This command targets production resources. Verify before proceeding." >&2
        # Still allow (exit 0) but Claude sees the warning
    fi
fi

exit 0
