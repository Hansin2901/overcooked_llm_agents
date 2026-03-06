# Observability & Tracing Guide

The Overcooked-AI project supports two complementary tracing systems for monitoring and debugging LLM agent behavior:

## Overview

| Feature | LangFuse (Custom) | LangSmith (LangChain) |
|---------|-------------------|----------------------|
| **Token Usage Tracking** | ✅ Detailed | ✅ Basic |
| **Cost Estimation** | ✅ Custom rates | ✅ Standard rates |
| **Custom Event Logging** | ✅ Full control | ⚠️ Limited |
| **LangGraph Visualization** | ❌ Manual spans | ✅ Automatic |
| **Setup Complexity** | Medium | Easy |
| **Best For** | Production metrics, cost tracking | Development, debugging |

**Recommendation**: Use **both systems simultaneously** for comprehensive observability.

---

## 1. LangSmith Tracing (Recommended for Development)

**LangSmith** provides automatic tracing for all LangChain/LangGraph operations with zero code changes.

### Setup

1. **Create Account**: Sign up at [smith.langchain.com](https://smith.langchain.com) (free, no credit card required)

2. **Get API Key**: Go to [Settings → API Keys](https://smith.langchain.com/settings) → Create API Key

3. **Configure Environment Variables**:

```bash
# In your .env file:
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...  # Your API key from step 2
LANGCHAIN_PROJECT=overcooked-ai  # Optional: project name (defaults to "default")
```

4. **Run Your Agent** (tracing happens automatically):

```bash
source .env && uv run python scripts/run_llm_agent.py \
  --agent-type planner-worker \
  --layout cramped_room \
  --horizon 20 \
  --debug
```

5. **View Traces**: Visit [smith.langchain.com/projects](https://smith.langchain.com/projects) to see your traces

### What You Get

- **Automatic Trace Capture**: Every LLM call, tool invocation, and graph execution is automatically logged
- **Visual Execution Flow**: See the exact path your agent took through the graph
- **Latency Tracking**: Identify slow operations
- **Token Usage**: Track input/output tokens per call
- **Error Debugging**: Automatic capture of exceptions with full context

### Example Trace View

```
Run: planner-worker-cramped_room-20260305-143022
├─ [Step 0] worker_0
│  ├─ llm_api_call (245ms)
│  │  ├─ Input: 1,234 tokens
│  │  ├─ Output: 45 tokens
│  │  └─ Model: moonshotai.kimi-k2.5
│  ├─ Tool: get_surroundings
│  └─ Tool: move_up
├─ [Step 0] worker_1
│  ├─ llm_api_call (312ms)
│  └─ Tool: move_left
└─ [Step 5] planner
   ├─ llm_api_call (567ms)
   └─ Tool: assign_tasks
```

---

## 2. LangFuse Tracing (Recommended for Production)

**LangFuse** is already integrated into the codebase and provides detailed custom metrics including cost estimation for custom models.

### Setup

1. **Create Account**: Sign up at [cloud.langfuse.com](https://cloud.langfuse.com) or [us.cloud.langfuse.com](https://us.cloud.langfuse.com)

2. **Get API Keys**: Go to Settings → API Keys

3. **Configure Environment Variables**:

```bash
# In your .env file:
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com  # Or https://cloud.langfuse.com for EU
```

4. **Run Your Agent**:

```bash
source .env && uv run python scripts/run_llm_agent.py \
  --agent-type planner-worker \
  --layout cramped_room \
  --horizon 20 \
  --debug
```

### What You Get

- **Custom Cost Tracking**: Pre-configured cost rates for TritonAI and other custom models
- **Structured Event Logging**: Custom events for `llm.generation`, `tool.call`, `action.commit`, `planner.assignment`
- **Hierarchical Traces**: Nested spans for runs → steps → roles → events
- **Local JSONL Logs**: All events also written to `logs/agent_runs/*.jsonl` for offline analysis

### Custom Model Cost Rates

The system includes pre-configured cost rates (in USD per 1M tokens) for:

```python
{
    "api-gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "moonshotai.kimi-k2.5": {"input": 0.60, "output": 3.03},
    "api-mistral-small-3.2-2506": {"input": 0.05, "output": 0.18},
    # ... and more
}
```

See `src/overcooked_ai_py/agents/llm/observability.py` for the full list.

---

## 3. Using Both Systems Simultaneously

**Recommended configuration** for comprehensive observability:

```bash
# .env file
# LangSmith for development/debugging
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_...
LANGCHAIN_PROJECT=overcooked-ai

# LangFuse for production metrics/costs
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

### Benefits of Dual Tracing

1. **Development**: Use LangSmith for interactive debugging and visual inspection
2. **Production**: Use LangFuse for detailed cost tracking and custom metrics
3. **Validation**: Cross-reference both systems to ensure complete observability
4. **Zero Code Changes**: Both systems activate automatically via environment variables

---

## 4. Local JSONL Logging (Always Active)

Regardless of LangSmith/LangFuse configuration, all events are **always** written to local JSONL files:

```
logs/agent_runs/<run_id>.jsonl
```

Each line is a JSON object with:

```json
{
  "ts": "2026-03-05T14:30:22.123456+00:00",
  "run_id": "a1b2c3d4e5f6...",
  "run_name": "planner-worker-cramped_room-20260305-143022",
  "event_type": "llm.generation",
  "mode": "planner-worker",
  "layout": "cramped_room",
  "model": "moonshotai.kimi-k2.5",
  "step": 0,
  "agent_role": "worker_0",
  "payload": {
    "content": "I need to get an onion first...",
    "tool_call_count": 2,
    "prompt_tokens": 1234,
    "completion_tokens": 45,
    "estimated_cost_usd": 0.000876,
    "latency_ms": 245
  }
}
```

### Analyzing Local Logs

Use the provided analysis script:

```bash
uv run python scripts/analyze_performance.py logs/agent_runs/<run_id>.jsonl
```

---

## 5. Disabling Tracing

### Disable LangSmith

```bash
# Option 1: Remove from .env
# LANGCHAIN_TRACING_V2=true

# Option 2: Set to false
export LANGCHAIN_TRACING_V2=false
```

### Disable LangFuse

```bash
# Remove keys from .env:
# LANGFUSE_PUBLIC_KEY=...
# LANGFUSE_SECRET_KEY=...
```

Local JSONL logging **cannot be disabled** and is always active.

---

## 6. Troubleshooting

### LangSmith Not Working

**Symptom**: No traces appear in LangSmith dashboard

**Solutions**:

1. **Check Environment Variables**:
   ```bash
   python3 -c "import os; print('LANGCHAIN_TRACING_V2:', os.getenv('LANGCHAIN_TRACING_V2')); print('LANGCHAIN_API_KEY:', 'SET' if os.getenv('LANGCHAIN_API_KEY') else 'NOT SET')"
   ```

2. **Verify API Key**: Ensure your API key is valid at [smith.langchain.com/settings](https://smith.langchain.com/settings)

3. **Check Project Exists**: If using `LANGCHAIN_PROJECT`, verify the project exists in your LangSmith account

4. **Network Access**: Ensure your environment can reach `https://api.smith.langchain.com`

5. **Load .env File**: Make sure you're using `source .env &&` before running scripts

### LangFuse Not Working

**Symptom**: No traces appear in LangFuse dashboard, but local JSONL logs work

**Solutions**:

1. **Check Environment Variables**:
   ```bash
   python3 -c "import os; print('LANGFUSE_PUBLIC_KEY:', 'SET' if os.getenv('LANGFUSE_PUBLIC_KEY') else 'NOT SET'); print('LANGFUSE_SECRET_KEY:', 'SET' if os.getenv('LANGFUSE_SECRET_KEY') else 'NOT SET')"
   ```

2. **Verify API Keys**: Check your keys at [cloud.langfuse.com/settings](https://cloud.langfuse.com/settings) or [us.cloud.langfuse.com/settings](https://us.cloud.langfuse.com/settings)

3. **Check Host URL**: Ensure `LANGFUSE_HOST` matches your account region (US vs EU)

4. **Debug Output**: Run with `--debug` flag to see observability errors:
   ```bash
   uv run python scripts/run_llm_agent.py --debug ...
   ```

5. **Langfuse Package**: Verify `langfuse` is installed:
   ```bash
   uv pip list | grep langfuse
   ```

### Local JSONL Logs Not Created

**Symptom**: No files in `logs/agent_runs/`

**Solutions**:

1. **Check Directory Permissions**: Ensure the script can create the `logs/agent_runs/` directory

2. **Verify Run Started**: The file is created when `hub.emit("run.start", ...)` is called

3. **Check for Crashes**: If the script crashes before starting, no log file will be created

---

## 7. Performance Impact

### LangSmith

- **Overhead**: ~5-10ms per traced operation (network calls)
- **Best Practice**: Enable only during development/debugging
- **Production**: Set `LANGCHAIN_TRACING_V2=false` in production

### LangFuse

- **Overhead**: ~2-5ms per event (async background flushes)
- **Best Practice**: Safe for production use
- **Graceful Degradation**: Failures don't crash the agent

### Local JSONL

- **Overhead**: <1ms per event (local file write)
- **Best Practice**: Always enabled (production-safe)

---

## 8. API Reference

### ObservabilityHub

The central observability coordinator in `src/overcooked_ai_py/agents/llm/observability.py`:

```python
from overcooked_ai_py.agents.llm.observability import (
    ObservabilityHub,
    FileRunLogger,
    LangFuseReporter,
    build_run_context,
)

# Create observability hub
context = build_run_context(args, mode="planner-worker", layout="cramped_room", model="gpt-4o")
file_logger = FileRunLogger(base_dir="logs/agent_runs", context=context)
langfuse = LangFuseReporter(enabled=True, context=context)
hub = ObservabilityHub(file_logger=file_logger, langfuse=langfuse)

# Start/end run
hub.start_run()
hub.end_run({"total_reward": 10, "steps": 50})

# Step lifecycle
hub.start_step(0)
hub.start_role("worker_0")
hub.emit("llm.generation", {"content": "...", "latency_ms": 245})
hub.end_role()
hub.end_step()
```

### Event Types

| Event Type | Description | Payload Keys |
|------------|-------------|--------------|
| `run.start` | Episode begins | `horizon` |
| `run.end` | Episode completes | `total_reward`, `steps`, `elapsed_s` |
| `llm.generation` | LLM API call | `content`, `model_name`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `estimated_cost_usd` |
| `tool.call` | Tool invoked | `tool_name`, `args` |
| `action.commit` | Agent commits action | `action`, `reasoning` |
| `planner.assignment` | Planner assigns tasks | `tasks` (dict of worker_id -> task) |
| `error` | Error occurred | `where`, `message` |

---

## 9. Best Practices

### Development Workflow

1. **Enable Both Systems**: Get comprehensive visibility
2. **Use `--debug` Flag**: See LLM reasoning in real-time
3. **Check LangSmith First**: Visual debugging is faster than reading logs
4. **Reference JSONL for Details**: When you need exact token counts or custom metrics

### Production Workflow

1. **Disable LangSmith**: Reduce latency overhead (`LANGCHAIN_TRACING_V2=false`)
2. **Keep LangFuse Enabled**: Track costs and performance metrics
3. **Monitor JSONL Logs**: Set up log rotation and aggregation
4. **Alert on Errors**: Parse JSONL for `"event_type": "error"`

### Cost Optimization

1. **Sample Traces**: Don't trace every request in high-volume production
2. **Filter by Layout**: Only trace specific layouts during development
3. **Use Short Horizons**: Test with `--horizon 10` for faster iteration
4. **Monitor Token Usage**: Check cost estimates in LangFuse

---

## 10. Further Reading

- **LangSmith Docs**: https://docs.smith.langchain.com/
- **LangFuse Docs**: https://langfuse.com/docs
- **LangGraph Tracing**: https://langchain-ai.github.io/langgraph/how-tos/tracing/
- **LangChain Callbacks**: https://python.langchain.com/docs/concepts/callbacks/

---

## Quick Reference: Environment Variables

```bash
# === LangSmith (LangChain Built-in) ===
LANGCHAIN_TRACING_V2=true                    # Enable tracing
LANGCHAIN_API_KEY=lsv2_pt_...                # API key from smith.langchain.com
LANGCHAIN_PROJECT=overcooked-ai              # Optional: project name
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com  # Optional: custom endpoint

# === LangFuse (Custom System) ===
LANGFUSE_PUBLIC_KEY=pk-lf-...                # Public key
LANGFUSE_SECRET_KEY=sk-lf-...                # Secret key
LANGFUSE_HOST=https://us.cloud.langfuse.com  # Or https://cloud.langfuse.com

# === Local Logging (Always Active) ===
# No configuration needed - logs always written to logs/agent_runs/
```
