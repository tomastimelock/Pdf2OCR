# DigitalOcean Claude Code Module -- Complete Setup Report

## Part I: Holistic Overview

### What This Module Is

The `digitalocean-claude/` directory is a self-contained, portable module that turns Claude Code into a DigitalOcean infrastructure operator. It is not a Python library, not an npm package, and not a standalone application. It is a **configuration package** built on four Claude Code extension mechanisms that work in concert:

1. **Memory & Rules** -- persistent instructions Claude loads at session start
2. **Skills** -- slash-command prompts that teach Claude domain-specific workflows
3. **Sub-Agents** -- isolated AI workers with their own model, tools, and system prompt
4. **MCP Server** -- a long-running Python process that exposes DigitalOcean API operations as native Claude tools

Together, these four mechanisms create a layered system where Claude knows *what* DigitalOcean is (memory), *how* to operate each service (skills), *who* should handle each kind of task (agents), and *can call the API directly* without shell commands (MCP server). The module is designed so that copying the folder into any project root immediately gives that project full DigitalOcean integration with no code changes.

### How the Layers Interact

```
User prompt
    |
    v
CLAUDE.md + rules/digitalocean.md        <-- Layer 1: Memory
    |  Loaded at session start. Establishes naming conventions,
    |  safety rules, tagging policy, and region defaults.
    |  Always active. Shapes every response.
    |
    v
Skills (do-droplets, do-dns, ...)         <-- Layer 2: Skills
    |  Loaded on-demand when Claude detects relevance or
    |  when the user types /do-droplets, /do-dns, etc.
    |  Inject domain expertise: exact CLI commands, size
    |  tables, best practices, worked examples.
    |
    v
Agents (do-manager, do-monitor, do-deploy) <-- Layer 3: Agents
    |  Claude delegates to these when the task is complex.
    |  Each runs in its own context window with a focused
    |  system prompt, restricted tool set, and chosen model.
    |
    v
MCP Server (digitalocean_mcp.py)          <-- Layer 4: MCP
    |  A stdio process that speaks JSON-RPC. Exposes 21
    |  tools that call the DigitalOcean REST API directly.
    |  Operates independently of doctl.
    |
    v
settings.json + hook scripts              <-- Cross-cutting: Security
       Permissions (allow/ask/deny) gate every Bash command.
       PreToolUse hooks validate commands before execution.
       .gitignore prevents secret leakage.
```

### Security Model

The module implements defence in depth across three independent enforcement points:

| Layer | Mechanism | What it stops |
|-------|-----------|---------------|
| **Permissions** (settings.json) | `allow` / `ask` / `deny` arrays using Claude Code's permission rule syntax with prefix matching | Read-only `doctl` commands auto-allowed; create/delete commands require user confirmation; `doctl auth` and `rm -rf` hard-denied; `.env` file reads blocked |
| **Hooks** (validate-do-command.sh) | Bash PreToolUse hook, runs before every Bash tool call, receives JSON on stdin, exits 0 (allow) or 2 (block) | Token exposure via echo/curl, mass tag-based deletion, auth credential changes. Production deletions pass but emit a stderr warning Claude surfaces |
| **MCP Server** (delete_droplet confirm flag) | Application-level boolean gate inside the tool handler | Droplet deletion without explicit `confirm=true` parameter |

A command must pass all three checks. A `doctl compute droplet delete` must survive the deny/ask rule in settings.json, not be blocked by the hook script, and -- if routed through MCP instead -- must include the confirm flag.

### Portability Design

The module is portable because it uses only relative paths and environment variable expansion. The `.mcp.json` references `mcp-server/digitalocean_mcp.py` relative to the project root. The `DIGITALOCEAN_API_TOKEN` is resolved via `${DIGITALOCEAN_API_TOKEN}` syntax at runtime. No absolute paths, no machine-specific configuration. Copy the folder, set the env var, start Claude Code.

---

## Part II: Fraction-Based File Analysis

### Directory Tree

```
digitalocean-claude/                      (20 files, 4 directories deep)
|
+-- .claude/                              Configuration root for Claude Code
|   +-- settings.json                     Permission rules & env vars
|   +-- CLAUDE.md                         Project memory (always loaded)
|   +-- rules/
|   |   +-- digitalocean.md               Path-conditional rules for *.tf, doctl, etc.
|   +-- skills/
|   |   +-- do-droplets/
|   |   |   +-- SKILL.md                  Droplet management skill (141 lines)
|   |   +-- do-kubernetes/
|   |   |   +-- SKILL.md                  Kubernetes management skill (167 lines)
|   |   +-- do-dns/
|   |   |   +-- SKILL.md                  DNS management skill (193 lines)
|   |   +-- do-spaces/
|   |   |   +-- SKILL.md                  Spaces/S3 management skill (232 lines)
|   |   +-- do-deploy/
|   |       +-- SKILL.md                  App Platform deployment skill (273 lines)
|   +-- agents/
|       +-- do-manager.md                 General infra agent (102 lines)
|       +-- do-monitor.md                 Monitoring agent (169 lines)
|       +-- do-deploy.md                  Deployment agent (240 lines)
|
+-- mcp-server/
|   +-- digitalocean_mcp.py              MCP server (659 lines)
|   +-- requirements.txt                 Python dependency (1 line)
|   +-- README.md                        MCP server documentation (67 lines)
|
+-- scripts/
|   +-- setup.sh                         Setup/verification script (143 lines)
|   +-- validate-do-command.sh           General command hook (45 lines)
|   +-- validate-deploy-command.sh       Deploy-agent command hook (37 lines)
|
+-- .mcp.json                            MCP server registration (11 lines)
+-- .env.example                         Environment variable template (11 lines)
+-- .gitignore                           Secret/artifact exclusions (18 lines)
```

---

### File-by-File Breakdown

---

#### `.claude/settings.json`

**Purpose:** The gatekeeper. Defines what Claude is allowed to do with Bash and Read tools in this project.

**Structure:** A single JSON object with two top-level keys: `permissions` and `env`.

**Permissions detail:**

| Category | Count | Pattern | Effect |
|----------|-------|---------|--------|
| `allow` | 24 rules | Read-only doctl commands (`list`, `get`, `logs`), file reads for `*.yaml`, `*.yml`, `*.tf`, `*.json`, `.env.example` | Auto-approved. No user prompt. Claude can freely list droplets, read Terraform files, check account info. |
| `ask` | 20 rules | Mutating doctl commands (`create`, `delete`, `update`, `node-pool`), deployment triggers | Claude must ask the user before executing. Covers droplets, kubernetes, domains, firewalls, volumes, databases, and apps. |
| `deny` | 4 rules | `doctl auth:*`, `rm -rf:*`, `Read(.env)`, `Read(**/secrets/**)` | Hard-blocked. No override. Prevents credential manipulation, filesystem destruction, and secret file exposure. |

**Environment injection:** Sets `DO_DEFAULT_REGION=nyc3` as a session environment variable available to all commands.

**Evaluation order:** Claude Code evaluates deny first, then ask, then allow. First match wins. A `doctl compute droplet delete` matches `ask` before it could match any `allow` rule.

---

#### `.claude/CLAUDE.md`

**Purpose:** The project brain. Loaded into Claude's context at every session start. Everything here shapes Claude's behavior for the entire session.

**Content sections:**

1. **Authentication** -- Tells Claude how the user authenticates (env var or `doctl auth init`). Claude can reference this when troubleshooting auth errors.

2. **Available Skills** -- Lists the five slash commands (`/do-droplets`, `/do-kubernetes`, `/do-dns`, `/do-spaces`, `/do-deploy`). This is how Claude knows what skills exist without scanning the filesystem.

3. **Available Agents** -- Lists the three agents (`do-manager`, `do-monitor`, `do-deploy`). Claude uses these names when delegating via the Task tool.

4. **Project Conventions** -- Defines the naming format `{project}-{environment}-{service}-{index}` with concrete examples. Defines four required tags: `project`, `environment`, `owner`, `created-by`. Defines region preferences (nyc3 primary, sfo3 secondary, ams3 for Europe).

5. **Common Workflows** -- Three numbered checklists for droplet creation, app deployment, and scaling. Claude follows these step sequences when asked to perform these operations.

6. **Safety Rules** -- Five imperative rules in all-caps emphasis. These are the behavioral constraints Claude follows regardless of what else is in context. The strongest: "NEVER delete production resources without explicit user confirmation."

7. **File References** -- Uses the `@.env.example` and `@scripts/setup.sh` import syntax. Claude Code resolves these at load time, pulling the content of those files into the memory context. This means Claude always knows what environment variables are needed and how the setup script works.

---

#### `.claude/rules/digitalocean.md`

**Purpose:** Conditional rules that activate only when Claude is working with specific file types.

**Activation:** The YAML frontmatter `paths` field triggers these rules when Claude touches files matching any of six glob patterns: `**/*.tf`, `**/*.tfvars`, `**/doctl*`, `**/*digitalocean*`, `**/app.yaml`, `**/app-spec.yaml`. When Claude is editing a Terraform file or an app spec, these rules load automatically. When Claude is editing a Python script, they do not.

**Content sections:**

1. **Resource Naming** -- Reinforces the `{project}-{environment}-{type}-{index}` convention with three concrete examples.

2. **Required Tags** -- Four mandatory tags with a twist: this file uses `managed-by` (terraform, claude-code, manual) whereas CLAUDE.md uses `created-by`. Both apply; they serve different tracking purposes.

3. **Region Selection** -- Expands beyond CLAUDE.md by including full Asia options (sgp1, blr1) and additional European options (fra1, lon1).

4. **Sizing Guidelines** -- Environment-specific recommendations for droplets (dev: s-1vcpu-1gb, prod: s-4vcpu-8gb+) and Kubernetes (dev: 1-2 nodes, prod: 3+ with auto-scaling).

5. **Security Requirements** -- Four numbered rules: SSH keys only, firewalls for public-facing resources, VPC for internal comms, monitoring on production.

6. **Terraform Best Practices** -- Five rules specific to HCL files: use digitalocean_project, remote state in Spaces, variables for secrets, consistent tagging, data sources for existing resources.

7. **App Platform Specifications** -- Five rules for app.yaml files: env vars for config, health checks, appropriate sizes, deploy-on-push, managed databases.

8. **Cost Awareness** -- Four directives: mention costs when creating, suggest cheaper alternatives, warn about expensive operations, consider reserved instances.

---

#### `.claude/skills/do-droplets/SKILL.md`

**Purpose:** Teaches Claude everything it needs to manage DigitalOcean virtual machines.

**Frontmatter:**
- `name: do-droplets` -- Invokable as `/do-droplets`
- `description` -- Claude uses this text to auto-detect when to load the skill. The phrase "Use when working with DO compute infrastructure" is the trigger signal.
- `allowed-tools: Bash(doctl:*), Read, Write` -- When this skill is active, Claude can run any doctl command and read/write files without per-use permission prompts.

**Body structure:**

The skill is organized as a reference manual, not a task script. It contains:

- **6 command templates** with exact `doctl` syntax for list, get, create, delete, resize, and actions (reboot, power-cycle, snapshot). Each includes all relevant flags.
- **Size table** -- 7 rows mapping slug to vCPUs, RAM, and monthly price ($4-$96/mo). This lets Claude recommend sizes by budget.
- **Image table** -- 6 rows covering Ubuntu 24.04/22.04, Debian 12, Rocky 9, AlmaLinux 9, Fedora 40.
- **Region table** -- 9 entries across 4 continents.
- **Helper commands** -- 4 commands for listing SSH keys, sizes, images, and regions. These are prerequisite lookups Claude runs before creating droplets.
- **Best practices** -- 6 numbered items (SSH keys, monitoring, VPC, tags, backups, --wait flag).
- **Worked example** -- A complete production web server creation script that chains SSH key lookup into droplet creation with all recommended flags.

---

#### `.claude/skills/do-kubernetes/SKILL.md`

**Purpose:** Kubernetes cluster lifecycle management on DigitalOcean.

**Frontmatter:**
- `allowed-tools: Bash(doctl:*), Bash(kubectl:*), Read, Write` -- Notably grants `kubectl` access in addition to `doctl`. This is the only skill that does so.

**Body structure:**

- **Cluster operations** -- Create, get, delete, kubeconfig save, version listing. The create command includes an inline node-pool spec with auto-scaling.
- **Node pool management** -- 5 operations: list, create, update (scale), delete, recycle. The create example includes min/max nodes for auto-scaling.
- **Cluster upgrades** -- Check available upgrades and upgrade commands.
- **Node size table** -- 6 entries including general-purpose (s-*), memory-optimized (g-*), and CPU-optimized (c-*) types.
- **kubectl integration** -- 7 kubectl commands for post-connection verification, workload inspection, and resource monitoring. This section only appears here because kubectl requires an active kubeconfig context.
- **Container registry** -- Create, login, and cluster attachment commands. Ties the registry to the cluster for private image pulls.
- **Worked example** -- Multi-node-pool production cluster with two pools (general + workers) using different sizes and auto-scaling ranges.

---

#### `.claude/skills/do-dns/SKILL.md`

**Purpose:** Domain and DNS record management.

**Frontmatter:**
- `allowed-tools: Bash(doctl:*), Read` -- No Write access. DNS operations are read + API calls, not file creation. This is deliberately more restricted than other skills.

**Body structure:**

- **Domain operations** -- 4 commands: list, create, delete, get.
- **DNS record operations** -- 4 commands: list, create, update, delete. The create command includes all flags (type, name, data, TTL).
- **Record type reference table** -- 8 types (A, AAAA, CNAME, MX, TXT, NS, SRV, CAA) with purpose descriptions and example data values.
- **7 complete configuration recipes:**
  - Point domain to droplet (with dynamic IP lookup via `--no-header`)
  - CNAME for subdomain
  - MX records with primary/secondary priorities
  - SPF TXT record
  - DKIM TXT record
  - DMARC TXT record
  - CAA record for Let's Encrypt
- **TTL guidelines table** -- 4 scenarios with recommended values (stable: 3600, migration: 300, etc.)
- **Verification commands** -- `dig` commands for checking A, MX, and TXT records, including querying specific nameservers.

This is the most recipe-heavy skill. DNS configuration involves many exact-value records (SPF syntax, DKIM format, DMARC policy strings) that are easy to get wrong. The recipes eliminate that risk.

---

#### `.claude/skills/do-spaces/SKILL.md`

**Purpose:** S3-compatible object storage management.

**Frontmatter:**
- `allowed-tools: Bash(doctl:*), Bash(s3cmd:*), Bash(aws:*), Read, Write` -- The broadest tool set of any skill. Grants access to three different CLI tools because Spaces can be managed via doctl, s3cmd, or the AWS CLI with a custom endpoint.

**Body structure:**

- **Three CLI paths** documented in parallel:
  - `doctl` -- Limited to CDN operations (Spaces creation isn't supported via doctl)
  - `s3cmd` -- Full operations: list, upload, download, delete, sync, ACL management. Includes `~/.s3cfg` configuration template.
  - `aws` CLI -- Same operations via `--endpoint-url` and `--profile` flags. Includes `aws configure` setup.
- **Regions table** -- 5 entries with full endpoint URLs.
- **CDN configuration** -- Create, list, and cache flush commands.
- **Access control** -- ACL commands (public/private) and a complete S3 bucket policy JSON example.
- **CORS configuration** -- Complete JSON spec and `aws s3api put-bucket-cors` application command.
- **3 use case recipes** -- Static website hosting, automated backup, and CDN-served application assets with resulting URL format.

---

#### `.claude/skills/do-deploy/SKILL.md`

**Purpose:** Application deployment to DigitalOcean App Platform.

**Frontmatter:**
- `disable-model-invocation: true` -- This is the only skill with this flag. Claude will **never** auto-load this skill. The user must explicitly type `/do-deploy`. This is intentional: deployments have real-world side effects and should only happen when deliberately triggered.

**Body structure:**

- **App operations** -- List, get, create, update, delete.
- **Deployment operations** -- Trigger, list, get, view logs (with component and type filters).
- **4 complete app spec YAML templates:**
  - Basic web service (Node.js with env vars and secrets)
  - Static site (Vite frontend with build output directory)
  - Worker (Python background job with queue URL secret)
  - Full-stack app (API + frontend + managed PostgreSQL database, with health check and inter-component database URL injection via `${db.DATABASE_URL}`)
- **Instance size table** -- 7 entries from basic-xxs ($5/mo) to professional-m ($100/mo).
- **Environment variable taxonomy** -- Types (GENERAL vs SECRET) and scopes (BUILD_TIME, RUN_TIME, BUILD_AND_RUN_TIME).
- **Health check spec** -- YAML with all 6 parameters (path, delays, period, timeout, thresholds).
- **Alert spec** -- YAML for deployment failure, domain failure, and CPU utilization alerts.
- **Rollback procedure** -- 3-command sequence to list deployments, extract a previous spec, and redeploy it.

This is the longest skill at 273 lines and the most template-heavy. It functions as a deployment playbook.

---

#### `.claude/agents/do-manager.md`

**Purpose:** The generalist. Handles any DigitalOcean infrastructure task that doesn't fall into monitoring or deployment.

**Frontmatter:**
- `model: sonnet` -- Uses Sonnet for balanced capability and cost.
- `tools: Bash, Read, Grep, Glob` -- Full tool access including search. Can find files, read configuration, and run commands.
- No `permissionMode` override -- Inherits default permission checking. User is prompted for destructive operations.

**System prompt design:**

The prompt is structured as a **workflow specification**, not a knowledge dump. It defines:

1. **5 capability domains** -- Compute, Networking, Storage, Databases, Kubernetes. Each lists 3-4 specific tasks. This tells Claude what it should *offer to do*.
2. **4-phase workflow** -- Assess Current State, Plan Changes, Execute with Confirmation, Verify and Report. This is prescriptive: Claude follows these phases in order.
3. **Command reference** -- 8 list commands and 2 account commands as quick-reference.
4. **Safety guidelines** -- 6 rules, the strongest being "NEVER delete production resources without explicit confirmation."
5. **Best practices** -- 6 numbered items covering naming, tagging, regions, security, monitoring, backups.

---

#### `.claude/agents/do-monitor.md`

**Purpose:** The observer. Read-only operations focused on health, metrics, and troubleshooting.

**Frontmatter:**
- `model: haiku` -- Uses the cheapest, fastest model. Monitoring queries are frequent, formulaic, and don't need deep reasoning. This is a deliberate cost optimization.
- `tools: Bash, Read, Grep` -- No Write, no Edit, no Glob. This agent cannot modify files. It can only run commands, read files, and search content.

**System prompt design:**

The prompt is structured as a **diagnostic manual**:

1. **4 focus areas** -- Resource health, metrics analysis, alert management, troubleshooting. Each has 3-4 sub-items.
2. **Command reference** -- 15 monitoring commands organized by category: CPU, memory, disk, bandwidth, load average, plus 4 status commands across resource types.
3. **Alert policy commands** -- List, create (with full flag set), and delete.
4. **Critical threshold table** -- 4 metrics with warning and critical values (CPU: 70%/85%, Memory: 75%/90%, Disk: 80%/90%, Load: cores*0.7/cores).
5. **Troubleshooting workflow** -- 4 phases: Identify Symptoms, Gather Data, Analyze, Recommend.
6. **4 common issue playbooks** -- High CPU, high memory, disk full, network issues. Each has 3 diagnostic steps.
7. **Quick health check** -- A single compound command that checks all resource types in one pass.

---

#### `.claude/agents/do-deploy.md`

**Purpose:** The deployer. Handles both App Platform and Kubernetes deployments.

**Frontmatter:**
- `model: sonnet` -- Needs Sonnet-level reasoning for deployment decisions.
- `tools: Bash, Read, Write, Glob` -- Can create files (app specs, rollback YAML) unlike the monitor agent.
- `permissionMode: acceptEdits` -- File edits are auto-accepted without user prompt. This is set because deployment workflows often generate temporary spec files and the constant prompting would be disruptive.

**System prompt design:**

The prompt is structured as a **deployment runbook**:

1. **4 responsibility areas** -- App Platform, container deployments, deployment operations, configuration management.
2. **7-item pre-deployment checklist** -- Uses checkbox format. Claude is expected to verify each item before proceeding.
3. **Dual-track command reference** -- App Platform commands (doctl apps) and Kubernetes commands (kubectl) in parallel sections. This agent handles both deployment targets.
4. **5-phase deployment workflow** -- Validate, Deploy, Monitor, Verify, Rollback. Each phase has commands for both App Platform and Kubernetes.
5. **Environment variable management** -- App Platform YAML syntax and Kubernetes secret creation commands side by side.
6. **Scaling instructions** -- App Platform (instance_count in spec) and Kubernetes (kubectl scale + autoscale).
7. **Post-deployment verification** -- 5 checks: health endpoint, functionality, error rates, resource utilization, log review.

---

#### `mcp-server/digitalocean_mcp.py`

**Purpose:** The API bridge. A Python process that speaks the Model Context Protocol over stdio, translating JSON-RPC requests into DigitalOcean REST API calls.

**Architecture:**

```
Claude Code                      MCP Server                    DigitalOcean API
    |                                |                               |
    |-- JSON-RPC (stdin) ----------->|                               |
    |   {"method":"tools/call",      |                               |
    |    "params":{"name":           |-- HTTP GET/POST/DELETE ------>|
    |      "list_droplets"}}         |   Authorization: Bearer $TOK  |
    |                                |                               |
    |                                |<-- JSON response ------------|
    |<-- JSON-RPC (stdout) ---------|                               |
    |   {"result":{"content":        |                               |
    |     [{"type":"text","text":    |                               |
    |       "..."}]}}                |                               |
```

**Startup sequence:**

1. Checks for `DIGITALOCEAN_API_TOKEN` env var. Exits with error message if missing.
2. Makes a verification GET to `/v2/account`. Exits if authentication fails.
3. Enters main loop: reads one JSON line from stdin per iteration.

**Protocol implementation:**

| MCP Method | Handler | Response |
|------------|---------|----------|
| `initialize` | Returns server info (name, version) and capabilities (`tools: {listChanged: false}`) | Protocol handshake |
| `notifications/initialized` | Returns `None` (no response) | Acknowledgment |
| `tools/list` | Returns the `TOOLS` array (21 tool definitions) | Tool discovery |
| `tools/call` | Routes to `handle_tool_call()`, catches HTTPError and generic exceptions | Tool execution |
| Anything else | Returns JSON-RPC error code -32601 (Method not found) | Error |

**Tool inventory (21 tools):**

| Category | Tools | HTTP Methods Used |
|----------|-------|-------------------|
| Account | `get_account`, `get_balance` | GET |
| Droplets | `list_droplets` (with pagination + tag filter), `get_droplet`, `create_droplet`, `delete_droplet`, `droplet_action` | GET, POST, DELETE |
| SSH Keys | `list_ssh_keys` | GET |
| Reference | `list_regions`, `list_sizes` | GET |
| DNS | `list_domains`, `list_domain_records`, `create_domain_record` | GET, POST |
| Databases | `list_databases`, `get_database` | GET |
| Kubernetes | `list_kubernetes_clusters`, `get_kubernetes_cluster` | GET |
| Firewalls | `list_firewalls` | GET |
| Load Balancers | `list_load_balancers` | GET |
| Volumes | `list_volumes` | GET |
| Apps | `list_apps`, `get_app`, `list_app_deployments` | GET |

**Input validation:**

- `create_droplet` has 4 required fields (`name`, `region`, `size`, `image`) enforced by the JSON schema. 6 optional fields are included only when provided.
- `delete_droplet` requires `confirm=true`. Without it, the handler returns an error before making any API call.
- `droplet_action` accepts 6 action types via enum validation. `resize` requires `size`, `snapshot` requires `name`.

**Error handling:**

Two catch layers:
1. `requests.exceptions.HTTPError` -- Extracts status code and response body (tries JSON, falls back to text). Returns with `isError: true`.
2. Generic `Exception` -- Returns error message string with `isError: true`.

Both return valid JSON-RPC responses (not exceptions), so the MCP connection stays alive after errors.

**Dependencies:** `requests>=2.28.0` only. No MCP SDK, no framework. The protocol is implemented directly as line-delimited JSON over stdin/stdout.

---

#### `mcp-server/requirements.txt`

**Content:** `requests>=2.28.0`

Single dependency. The MCP server uses `requests` for HTTP calls to the DigitalOcean REST API. No MCP framework is needed because the stdio protocol is simple enough to implement directly.

---

#### `mcp-server/README.md`

**Purpose:** Standalone documentation for the MCP server. Covers installation, registration with Claude Code, and a categorized list of all 21 tools.

---

#### `scripts/setup.sh`

**Purpose:** First-run validation. Checks all prerequisites and guides the user through initial configuration.

**Execution flow:**

1. **doctl check** -- Uses `command -v doctl`. If found, prints version. If missing, prints install instructions for macOS (brew), Linux (snap), and Windows (scoop). Notes that doctl is optional when using the MCP server.

2. **API token check** -- If `$DIGITALOCEAN_API_TOKEN` is set:
   - Tries `doctl account get --access-token` if doctl is available
   - Falls back to `curl` with HTTP status check if doctl is missing
   - Reports authenticated email on success

   If not set:
   - Prints the token generation URL
   - Offers interactive `read -p` prompt
   - Exports token for current session and prints shell profile persistence command

3. **Python check** -- Tries `python3`, falls back to `python`. Reports version or prints error.

4. **requests check** -- Tries `import requests` in both python3 and python. Reports status.

5. **jq check** -- Required by hook scripts for JSON parsing. Prints install instructions for macOS (brew) and Linux (apt).

6. **Script permissions** -- `chmod +x` on all `.sh` files in the scripts directory.

7. **Summary** -- Prints the module directory tree and example commands.

---

#### `scripts/validate-do-command.sh`

**Purpose:** PreToolUse hook for general DigitalOcean command validation. Runs before every Bash tool call.

**Input:** Receives MCP hook JSON on stdin. Extracts `.tool_input.command` using `jq`.

**Validation rules:**

| Check | Detection | Exit Code | Behavior |
|-------|-----------|-----------|----------|
| Token exposure | Command contains `DIGITALOCEAN_API_TOKEN` or `DO_API_TOKEN` AND contains `echo`, `print`, `cat`, or `curl.*Bearer` | 2 (block) | Prevents accidental token printing |
| Mass deletion | Command contains `doctl.*delete.*--force` AND `--tag-name` | 2 (block) | Prevents wiping all resources with a tag in one command |
| Auth modification | Command matches `doctl auth (init\|switch\|revoke)` | 2 (block) | Prevents credential changes through Claude |
| Production deletion | Command contains `doctl.*(delete\|destroy)` AND `(prod\|production)` | 0 (allow) with stderr warning | Lets it through but ensures Claude surfaces the warning |
| Everything else | No match | 0 (allow) | Default pass-through |

---

#### `scripts/validate-deploy-command.sh`

**Purpose:** PreToolUse hook scoped to the `do-deploy` agent. Referenced in the agent's frontmatter `hooks` field.

**Validation rules:**

| Check | Detection | Exit Code |
|-------|-----------|-----------|
| Destructive git | `git.*(push.*--force\|reset.*--hard\|clean.*-f)` | 2 (block) |
| Database mutations | `(DROP\|TRUNCATE\|DELETE FROM)` | 2 (block) |
| System destruction | `(rm -rf /\|shutdown\|reboot\|halt)` | 2 (block) |
| Everything else | No match | 0 (allow) |

This hook is narrower than the general one. It focuses on operations that would be catastrophic *during* a deployment: force-pushing code, dropping database tables, or shutting down the server being deployed to.

---

#### `.mcp.json`

**Purpose:** Project-level MCP server registration. Claude Code reads this at session start and launches the server process.

**Content:**
```json
{
  "mcpServers": {
    "digitalocean": {
      "command": "python",
      "args": ["mcp-server/digitalocean_mcp.py"],
      "env": {
        "DIGITALOCEAN_API_TOKEN": "${DIGITALOCEAN_API_TOKEN}"
      }
    }
  }
}
```

**Key details:**
- `command: "python"` -- Uses whatever `python` is on the PATH. On systems where only `python3` exists, this needs adjustment.
- `args` -- Relative path from project root.
- `env` -- Uses `${DIGITALOCEAN_API_TOKEN}` expansion syntax. Claude Code resolves this from the process environment at server startup. If the variable is not set, the MCP server's own validation catches it and exits with an error message.

---

#### `.env.example`

**Purpose:** Template for required and optional environment variables. Not loaded by Claude Code directly -- it exists as documentation and is imported into CLAUDE.md via the `@.env.example` syntax.

**Variables:**
- `DIGITALOCEAN_API_TOKEN` -- Required. Used by both the MCP server and doctl.
- `SPACES_ACCESS_KEY` / `SPACES_SECRET_KEY` -- Optional. Used by s3cmd/AWS CLI for object storage.
- `DO_DEFAULT_REGION` -- Optional. Convenience default.

---

#### `.gitignore`

**Purpose:** Prevents secrets and artifacts from entering version control.

**Rules:**
- `.env` and `.env.local` -- Actual credential files
- `.claude/*.local.*` and `CLAUDE.local.md` -- Per-user Claude preferences
- `__pycache__/`, `*.pyc`, `.venv/`, `venv/` -- Python artifacts
- `.DS_Store`, `Thumbs.db` -- OS metadata

---

## Part III: How the Parts Compose

### Scenario: User says "Create a staging droplet for the API"

1. **CLAUDE.md** is already in context. Claude knows the naming convention is `{project}-{env}-{service}-{index}` and the default region is `nyc3`.

2. Claude detects the request matches the `do-droplets` skill description ("Use when working with DO compute infrastructure"). The skill loads into context.

3. Claude now has the size table, image table, and the worked example. It decides to:
   - Look up SSH keys: `doctl compute ssh-key list` (auto-allowed by settings.json)
   - Create the droplet: `doctl compute droplet create api-staging-droplet-01 --region nyc3 --size s-2vcpu-2gb --image ubuntu-22-04-x64 ...` (triggers `ask` rule -- user must confirm)

4. Before the create command executes, `validate-do-command.sh` runs. The command doesn't match any block patterns. Exit 0.

5. Claude reports the result, suggests setting up a firewall (per the rules in `digitalocean.md` which say "Configure firewalls for all public-facing resources"), and applies the tags `project`, `environment:staging`, `created-by:claude-code`.

### Scenario: User says "Check if anything is unhealthy"

1. Claude recognizes this as a monitoring task and delegates to the `do-monitor` agent.

2. The agent starts in its own context window with Haiku model. It only has Bash, Read, and Grep tools.

3. The agent runs the "Quick Health Check" compound command from its system prompt. All commands are read-only `doctl` list commands, auto-allowed by settings.json.

4. Results return to the main conversation as a summary.

### Scenario: MCP server is active and user says "List my droplets"

1. Claude can use either doctl via Bash or the `list_droplets` MCP tool. The MCP tool is preferred because it doesn't require doctl to be installed.

2. Claude calls the MCP tool. The server makes `GET /v2/droplets` to the DigitalOcean API.

3. The JSON response is returned through the MCP protocol and Claude formats it for the user.
