---
title: DataGym
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
tags:
  - openenv
---

# DataGym — Data Cleaning & Wrangling Environment

An OpenEnv environment that trains AI agents to clean and transform messy real-world data by writing Python/pandas code. The agent's DataFrame is compared cell-by-cell against a known clean ground truth.

## Why DataGym?

Data cleaning is one of the most common, time-consuming tasks in real-world data work — data scientists spend [up to 80% of their time](https://hbr.org/2018/04/if-your-data-is-bad-your-machine-learning-tools-are-useless) preparing data. DataGym provides a structured environment where agents learn to:

- Fix numeric formats (dollar signs, commas, percentages)
- Handle null/missing values with appropriate strategies
- Standardize dates, text casing, and categories
- Parse semi-structured data (logs, JSON, compound fields)
- Perform multi-step data pipelines combining all of the above

## Action & Observation Spaces

### Action Space

The agent submits Python/pandas code as a string:

```python
DataAction(code="df['price'] = df['price'].str.replace('$', '').astype(float)")
```

Code executes in a sandboxed namespace with `df` (current DataFrame), `pd`, `np`, `re`, and `json` available. Dangerous operations (`os`, `subprocess`, `eval`, `exec`, etc.) are blocked.

### Observation Space

After each action, the agent receives:

| Field | Description |
|-------|-------------|
| `task_description` | What needs to be cleaned |
| `data_preview` | First 20 rows of the current DataFrame |
| `data_shape` | (rows, cols) shape |
| `column_info` | Column names, dtypes, null counts |
| `issues_found` | Detected data quality issues |
| `target_schema` | Expected column names and types |
| `current_score` | Score vs ground truth (0.0–1.0) |
| `code_output` | stdout from last execution |
| `code_error` | Error message if code failed |
| `hint` | Hint for easy tasks only |

## Tasks (15 total)

### Easy (5 tasks, max 5 steps each)
| Task ID | Description |
|---------|-------------|
| `e1_fix_numeric_types` | Remove `$`, `,`, `%` symbols and convert to proper numeric types |
| `e2_handle_nulls` | Replace sentinel nulls (`N/A`, `null`, `--`) and fill missing values |
| `e3_standardize_dates` | Normalize mixed date formats to ISO 8601 (YYYY-MM-DD) |
| `e4_fix_text` | Strip whitespace, standardize casing across text columns |
| `e5_remove_duplicates` | Identify and remove exact duplicate rows |

### Medium (5 tasks, max 8 steps each)
| Task ID | Description |
|---------|-------------|
| `m1_split_and_clean` | Split compound columns (City, ST ZIP) + currency cleanup |
| `m2_dedup_merge` | Fuzzy deduplication by email, keep highest score, re-index |
| `m3_types_and_outliers` | Type conversion + outlier capping + absolute values |
| `m4_parse_logs` | Parse structured log entries into timestamp/level/message |
| `m5_wide_to_long` | Pivot wide-format grades into long format |

### Hard (5 tasks, max 12 steps each)
| Task ID | Description |
|---------|-------------|
| `h1_full_pipeline` | End-to-end: dedup + dates + nulls + currency + casing |
| `h2_cross_table` | Separate embedded tables, fix names, recalculate totals, join |
| `h3_json_extract` | Parse JSON strings into structured columns with derived fields |
| `h4_timeseries_clean` | Deduplicate timestamps, fill gaps via linear interpolation |
| `h5_real_world` | Multi-issue HR dataset: dedup + names + dates + salary + ratings |

## Scoring

Each task is graded 0.0–1.0 via cell-level DataFrame comparison:

- **20%** Column structure — correct columns exist with right names
- **10%** Row count — correct number of rows
- **70%** Cell accuracy — individual cells matching expected values

Matching rules: numeric tolerance (1e-6), case-insensitive strings, whitespace-normalized, date-aware comparison. Safety violations incur a -0.1 penalty.

## Setup & Usage

### Prerequisites
- Python 3.10+
- Docker (for containerized deployment)

### Local Development

```bash
cd data_gym
python -m venv .venv && source .venv/bin/activate
pip install -e .
# or with uv:
uv sync

# Start the server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Verify endpoints
curl http://localhost:8000/health
curl http://localhost:8000/tasks
```

### Docker

```bash
# Build (using openenv CLI)
openenv build .

# Or build directly
docker build -t data-gym .

# Run
docker run -p 8000:8000 data-gym
```

### Run Baseline

```bash
# With OpenAI API
export OPENAI_API_KEY=sk-...
python baseline.py --env-url http://localhost:8000

# With local model (e.g., LM Studio)
python baseline.py \
  --base-url http://localhost:1234/v1 \
  --api-key lm-studio \
  --model qwen/qwen3-4b \
  --env-url http://localhost:8000
```

## Baseline Scores

### Golden Reference (deterministic code)

Achieves **1.000 average** across all 15 tasks — available via `POST /baseline`.

### LLM Baseline (qwen2.5:7b, 5 steps per task)

| Difficulty | Tasks | Avg Score | Pass Rate |
|-----------|-------|-----------|-----------|
| Easy | 5 | 1.000 | 5/5 |
| Medium | 5 | 0.607 | 2/5 |
| Hard | 5 | 0.570 | 1/5 |
| **Overall** | **15** | **0.726** | **8/15** |

Easy tasks are reliably solved in 1–2 steps. Medium tasks reward multi-step reasoning (m1 scored 0.996, m3 scored 0.986). Hard tasks provide genuine challenge — the model struggles with sandbox constraints (`import` is blocked; `pd`, `np`, `re`, `json` are pre-loaded) and complex multi-step pipelines.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List all tasks with action schema |
| `/grader` | GET | Grading methodology details |
| `/baseline` | POST | Run golden baseline, return scores |
| WebSocket | — | `reset()`, `step()`, `state()` via OpenEnv protocol |

## Project Structure

```
data_gym/
├── openenv.yaml          # OpenEnv spec metadata
├── models.py             # Pydantic models (Action, Observation, State)
├── client.py             # WebSocket client for agents
├── baseline.py           # LLM baseline inference script
├── Dockerfile            # Multi-stage container build
├── pyproject.toml        # Package config
└── server/
    ├── app.py            # FastAPI application + competition endpoints
    ├── data_gym_environment.py  # Core environment logic
    ├── grading.py        # Cell-level DataFrame grading engine
    └── tasks/
        ├── registry.py   # Task registration system
        ├── easy.py       # 5 easy tasks
        ├── medium.py     # 5 medium tasks
        └── hard.py       # 5 hard tasks
```

## Reward Design

- Continuous signal (0.0–1.0) on every step, not just end-of-episode
- Partial credit for partial progress (some columns right, some rows right)
- Safety penalty (-0.1) for forbidden operations
- Episode terminates on score >= 0.95 or max steps reached
- Failed episodes (score < 0.3) get halved reward to discourage flailing
- **Score breakdown** in `code_output` tells agents exactly which columns/cells are wrong, with examples of expected vs actual values
- **Progress tracking** shows whether each step improved, regressed, or left the score unchanged

## Sandbox Details

Code runs in a restricted Python sandbox. Key rules for agents:

- **No `import` statements** — `pd`, `np`, `re`, `json` are pre-loaded in scope
- **`df` is the working DataFrame** — modify in-place or assign to `result`
- Blocked: `os`, `subprocess`, `eval`, `exec`, `open`, `shutil`, `requests`, `socket`
- If `import` is attempted, the error message explicitly lists available modules

## Example Agent Interaction

```
RESET task=e1_fix_numeric_types
→ Observation: score=0.606, data has $29.99 prices, 1,000 quantities, 10% discounts

STEP code="df['price'] = df['price'].str.replace('$','',regex=False).str.strip().astype(float)"
→ score=0.756 (improved from 0.606)
→ breakdown: 'discount': 8 wrong, 'quantity': 4 wrong

STEP code="df['quantity'] = df['quantity'].str.replace(',','').astype(int)\ndf['discount'] = df['discount'].str.replace('%','').astype(float) / 100"
→ score=1.000 ✓ (done)
```
