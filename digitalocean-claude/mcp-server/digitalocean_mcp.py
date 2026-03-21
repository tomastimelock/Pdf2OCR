#!/usr/bin/env python3
"""
DigitalOcean MCP Server for Claude Code

A Model Context Protocol server that provides tools for managing
DigitalOcean resources. Runs as a stdio transport server.

Usage:
    python digitalocean_mcp.py

Requires:
    DIGITALOCEAN_API_TOKEN environment variable
"""

import json
import os
import sys
from typing import Any

try:
    import requests
except ImportError:
    print(
        "Error: 'requests' package is required. Install with: pip install requests",
        file=sys.stderr,
    )
    sys.exit(1)

DO_API_URL = "https://api.digitalocean.com/v2"
DO_TOKEN = os.environ.get("DIGITALOCEAN_API_TOKEN")


def get_headers() -> dict:
    return {
        "Authorization": f"Bearer {DO_TOKEN}",
        "Content-Type": "application/json",
    }


def do_get(path: str, params: dict = None) -> dict:
    """Make a GET request to the DigitalOcean API."""
    response = requests.get(
        f"{DO_API_URL}/{path}", headers=get_headers(), params=params
    )
    response.raise_for_status()
    return response.json()


def do_post(path: str, payload: dict) -> dict:
    """Make a POST request to the DigitalOcean API."""
    response = requests.post(
        f"{DO_API_URL}/{path}", headers=get_headers(), json=payload
    )
    response.raise_for_status()
    return response.json()


def do_delete(path: str) -> dict:
    """Make a DELETE request to the DigitalOcean API."""
    response = requests.delete(f"{DO_API_URL}/{path}", headers=get_headers())
    if response.status_code == 204:
        return {"success": True}
    response.raise_for_status()
    return response.json()


def do_put(path: str, payload: dict) -> dict:
    """Make a PUT request to the DigitalOcean API."""
    response = requests.put(
        f"{DO_API_URL}/{path}", headers=get_headers(), json=payload
    )
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    # Account
    {
        "name": "get_account",
        "description": "Get DigitalOcean account information including email, droplet limit, and status",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_balance",
        "description": "Get current account balance, month-to-date usage, and billing info",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # Droplets
    {
        "name": "list_droplets",
        "description": "List all DigitalOcean droplets with optional tag filter",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tag_name": {
                    "type": "string",
                    "description": "Filter droplets by tag name (optional)",
                },
                "page": {
                    "type": "integer",
                    "description": "Page number (default: 1)",
                },
                "per_page": {
                    "type": "integer",
                    "description": "Results per page, max 200 (default: 20)",
                },
            },
        },
    },
    {
        "name": "get_droplet",
        "description": "Get detailed information about a specific droplet by ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "droplet_id": {
                    "type": "integer",
                    "description": "The droplet ID",
                }
            },
            "required": ["droplet_id"],
        },
    },
    {
        "name": "create_droplet",
        "description": "Create a new DigitalOcean droplet (virtual machine)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Droplet hostname"},
                "region": {
                    "type": "string",
                    "description": "Region slug (e.g., nyc3, sfo3, ams3)",
                },
                "size": {
                    "type": "string",
                    "description": "Size slug (e.g., s-1vcpu-1gb, s-2vcpu-4gb)",
                },
                "image": {
                    "type": "string",
                    "description": "Image slug or ID (e.g., ubuntu-22-04-x64)",
                },
                "ssh_keys": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Array of SSH key IDs to add",
                },
                "backups": {
                    "type": "boolean",
                    "description": "Enable automated backups (default: false)",
                },
                "monitoring": {
                    "type": "boolean",
                    "description": "Enable monitoring agent (default: false)",
                },
                "vpc_uuid": {
                    "type": "string",
                    "description": "VPC UUID for private networking",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Tags to apply to the droplet",
                },
                "user_data": {
                    "type": "string",
                    "description": "Cloud-init user data script",
                },
            },
            "required": ["name", "region", "size", "image"],
        },
    },
    {
        "name": "delete_droplet",
        "description": "Delete a DigitalOcean droplet. This is destructive and cannot be undone.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "droplet_id": {
                    "type": "integer",
                    "description": "The droplet ID to delete",
                },
                "confirm": {
                    "type": "boolean",
                    "description": "Must be true to confirm deletion",
                },
            },
            "required": ["droplet_id", "confirm"],
        },
    },
    {
        "name": "droplet_action",
        "description": "Perform an action on a droplet (reboot, power_off, power_on, shutdown, snapshot, resize)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "droplet_id": {
                    "type": "integer",
                    "description": "The droplet ID",
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "reboot",
                        "power_off",
                        "power_on",
                        "shutdown",
                        "snapshot",
                        "resize",
                    ],
                    "description": "The action to perform",
                },
                "size": {
                    "type": "string",
                    "description": "New size slug (required for resize action)",
                },
                "name": {
                    "type": "string",
                    "description": "Snapshot name (required for snapshot action)",
                },
            },
            "required": ["droplet_id", "action"],
        },
    },
    # SSH Keys
    {
        "name": "list_ssh_keys",
        "description": "List all SSH keys in the account",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # Regions and Sizes
    {
        "name": "list_regions",
        "description": "List all available DigitalOcean regions",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_sizes",
        "description": "List all available droplet sizes with pricing",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # Domains and DNS
    {
        "name": "list_domains",
        "description": "List all domains in the account",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "list_domain_records",
        "description": "List all DNS records for a domain",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "description": "Domain name",
                }
            },
            "required": ["domain"],
        },
    },
    {
        "name": "create_domain_record",
        "description": "Create a DNS record for a domain",
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain": {"type": "string", "description": "Domain name"},
                "type": {
                    "type": "string",
                    "enum": ["A", "AAAA", "CNAME", "MX", "TXT", "NS", "SRV", "CAA"],
                    "description": "Record type",
                },
                "name": {
                    "type": "string",
                    "description": "Record name (@ for root)",
                },
                "data": {
                    "type": "string",
                    "description": "Record data (IP, hostname, etc.)",
                },
                "ttl": {
                    "type": "integer",
                    "description": "TTL in seconds (default: 3600)",
                },
                "priority": {
                    "type": "integer",
                    "description": "Priority (for MX and SRV records)",
                },
            },
            "required": ["domain", "type", "name", "data"],
        },
    },
    # Databases
    {
        "name": "list_databases",
        "description": "List all managed database clusters",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_database",
        "description": "Get details of a specific database cluster",
        "inputSchema": {
            "type": "object",
            "properties": {
                "database_id": {
                    "type": "string",
                    "description": "Database cluster ID",
                }
            },
            "required": ["database_id"],
        },
    },
    # Kubernetes
    {
        "name": "list_kubernetes_clusters",
        "description": "List all Kubernetes clusters",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_kubernetes_cluster",
        "description": "Get details of a specific Kubernetes cluster",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cluster_id": {
                    "type": "string",
                    "description": "Kubernetes cluster ID",
                }
            },
            "required": ["cluster_id"],
        },
    },
    # Firewalls
    {
        "name": "list_firewalls",
        "description": "List all firewalls",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # Load Balancers
    {
        "name": "list_load_balancers",
        "description": "List all load balancers",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # Volumes
    {
        "name": "list_volumes",
        "description": "List all block storage volumes",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # Apps
    {
        "name": "list_apps",
        "description": "List all App Platform applications",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_app",
        "description": "Get details of a specific App Platform application",
        "inputSchema": {
            "type": "object",
            "properties": {
                "app_id": {
                    "type": "string",
                    "description": "App ID",
                }
            },
            "required": ["app_id"],
        },
    },
    {
        "name": "list_app_deployments",
        "description": "List deployments for an App Platform application",
        "inputSchema": {
            "type": "object",
            "properties": {
                "app_id": {
                    "type": "string",
                    "description": "App ID",
                }
            },
            "required": ["app_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def handle_tool_call(name: str, arguments: dict) -> Any:
    """Route tool calls to the DigitalOcean API."""

    # Account
    if name == "get_account":
        return do_get("account")
    if name == "get_balance":
        return do_get("customers/my/balance")

    # Droplets
    if name == "list_droplets":
        params = {}
        if arguments.get("tag_name"):
            params["tag_name"] = arguments["tag_name"]
        if arguments.get("page"):
            params["page"] = arguments["page"]
        if arguments.get("per_page"):
            params["per_page"] = arguments["per_page"]
        return do_get("droplets", params)

    if name == "get_droplet":
        return do_get(f"droplets/{arguments['droplet_id']}")

    if name == "create_droplet":
        payload = {
            "name": arguments["name"],
            "region": arguments["region"],
            "size": arguments["size"],
            "image": arguments["image"],
        }
        for field in [
            "ssh_keys",
            "backups",
            "monitoring",
            "vpc_uuid",
            "tags",
            "user_data",
        ]:
            if arguments.get(field) is not None:
                payload[field] = arguments[field]
        return do_post("droplets", payload)

    if name == "delete_droplet":
        if not arguments.get("confirm"):
            return {"error": "Deletion not confirmed. Set confirm=true to proceed."}
        result = do_delete(f"droplets/{arguments['droplet_id']}")
        if result.get("success"):
            return {"message": f"Droplet {arguments['droplet_id']} deleted successfully"}
        return result

    if name == "droplet_action":
        action = arguments["action"]
        payload = {"type": action}
        if action == "resize" and arguments.get("size"):
            payload["size"] = arguments["size"]
        if action == "snapshot" and arguments.get("name"):
            payload["name"] = arguments["name"]
        return do_post(f"droplets/{arguments['droplet_id']}/actions", payload)

    # SSH Keys
    if name == "list_ssh_keys":
        return do_get("account/keys")

    # Regions and Sizes
    if name == "list_regions":
        return do_get("regions")
    if name == "list_sizes":
        return do_get("sizes")

    # Domains and DNS
    if name == "list_domains":
        return do_get("domains")
    if name == "list_domain_records":
        return do_get(f"domains/{arguments['domain']}/records")
    if name == "create_domain_record":
        payload = {
            "type": arguments["type"],
            "name": arguments["name"],
            "data": arguments["data"],
            "ttl": arguments.get("ttl", 3600),
        }
        if arguments.get("priority") is not None:
            payload["priority"] = arguments["priority"]
        return do_post(f"domains/{arguments['domain']}/records", payload)

    # Databases
    if name == "list_databases":
        return do_get("databases")
    if name == "get_database":
        return do_get(f"databases/{arguments['database_id']}")

    # Kubernetes
    if name == "list_kubernetes_clusters":
        return do_get("kubernetes/clusters")
    if name == "get_kubernetes_cluster":
        return do_get(f"kubernetes/clusters/{arguments['cluster_id']}")

    # Firewalls
    if name == "list_firewalls":
        return do_get("firewalls")

    # Load Balancers
    if name == "list_load_balancers":
        return do_get("load_balancers")

    # Volumes
    if name == "list_volumes":
        return do_get("volumes")

    # Apps
    if name == "list_apps":
        return do_get("apps")
    if name == "get_app":
        return do_get(f"apps/{arguments['app_id']}")
    if name == "list_app_deployments":
        return do_get(f"apps/{arguments['app_id']}/deployments")

    return {"error": f"Unknown tool: {name}"}


# ---------------------------------------------------------------------------
# MCP protocol handling
# ---------------------------------------------------------------------------

SERVER_INFO = {
    "name": "digitalocean",
    "version": "1.0.0",
    "description": "DigitalOcean infrastructure management server",
}


def handle_request(request: dict) -> dict:
    """Handle an incoming MCP JSON-RPC request."""
    method = request.get("method", "")
    req_id = request.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": SERVER_INFO,
                "capabilities": {"tools": {"listChanged": False}},
            },
        }

    if method == "notifications/initialized":
        return None  # No response needed for notifications

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS},
        }

    if method == "tools/call":
        tool_name = request["params"]["name"]
        arguments = request["params"].get("arguments", {})
        try:
            result = handle_tool_call(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(result, indent=2, default=str),
                        }
                    ]
                },
            }
        except requests.exceptions.HTTPError as e:
            error_body = ""
            if e.response is not None:
                try:
                    error_body = e.response.json()
                except Exception:
                    error_body = e.response.text
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "error": str(e),
                                    "status_code": (
                                        e.response.status_code
                                        if e.response is not None
                                        else None
                                    ),
                                    "details": error_body,
                                },
                                indent=2,
                                default=str,
                            ),
                        }
                    ],
                    "isError": True,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {"type": "text", "text": json.dumps({"error": str(e)})}
                    ],
                    "isError": True,
                },
            }

    # Unknown method
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main():
    """Main loop: read JSON-RPC from stdin, write responses to stdout."""
    if not DO_TOKEN:
        print(
            "Error: DIGITALOCEAN_API_TOKEN environment variable is not set.\n"
            "Get your token from: https://cloud.digitalocean.com/account/api/tokens",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify token works
    try:
        requests.get(f"{DO_API_URL}/account", headers=get_headers()).raise_for_status()
    except Exception as e:
        print(f"Error: Failed to authenticate with DigitalOcean API: {e}", file=sys.stderr)
        sys.exit(1)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue

        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
