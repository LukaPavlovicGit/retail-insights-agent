# Retail Insights Agent

A data-analysis chat assistant for non-technical retail executives. You ask questions about
sales, products, customers, returns and delivery in plain English. The agent writes BigQuery
SQL over the public `bigquery-public-data.thelook_ecommerce` dataset, checks and rewrites it
in code (SELECT only, no personal columns, limited to your brands), runs it
read-only, and explains the result. It can save answers as reports and delete reports after
you confirm. It is built with LangGraph, Gemini, an MCP tool server, and SQLite with
sqlite-vec.

## Architecture

- [docs/assignment.md](docs/assignment.md): the assignment brief this project answers.
- [docs/data-model.md](docs/data-model.md): the warehouse tables and their relationships, the agent's own store, and how each table is used.
- [docs/hld.md](docs/hld.md): production architecture on Google Cloud, the flow of one
  analysis turn, the delete confirmation flow, component choices, and non-functional notes.
- [docs/technical-design.md](docs/technical-design.md): one section per assignment
  requirement with the module and test that implement it, the error handling matrix, and
  the client's answers with what each one changed.

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- A Google Cloud project with the BigQuery API enabled (queries against the public dataset
  are billed to this project; the agent caps each query at 1 GB scanned)
- The `gcloud` CLI
- A Gemini API key from Google AI Studio

## Setup

```bash
git clone https://github.com/LukaPavlovicGit/retail-insights-agent
cd retail-insights-agent
uv sync
cp .env.example .env            # fill GEMINI_API_KEY and GOOGLE_CLOUD_PROJECT
gcloud auth application-default login
uv run ria doctor
uv run ria init
uv run ria chat --user alice    # or bob, carol (CEO, unrestricted)
```

If you upgrade from an older checkout, delete `ria.db` and run `ria init` again.

- `ria doctor` checks the Gemini key, Google credentials, a BigQuery dry run, and one Gemini
  call.
- `ria init` loads the demo users and the 15 seed Golden Trios into `ria.db`, embeds the
  Trio questions, caches the live BigQuery schema in `data/schema.json`, and checks that the
  brands in `data/users.yaml` exist in the dataset. Run it again after promoting a Trio.
- Demo users: `alice` (VP Denim and Workwear: Levi's, Carhartt, Dockers, Wrangler), `bob`
  (VP Outdoor and Swim: Columbia, Volcom, Quiksilver, Speedo), `carol` (CEO, all brands).
- `ria chat --thread <id>` resumes an earlier conversation; the thread id is printed when a
  chat starts.

## Chat commands

Commands are recognised in code before any model call, so they work even when the language
model is down. Type `/help` in a chat to see the list.

| Command | What it does |
|---|---|
| `/reports` | List your saved reports |
| `/report <id>` | Show one saved report |
| `/delete <filter>` | Delete reports ("from this conversation", "all", or "mentioning <subject>"); you confirm first |
| `/schema` | Show the tables and columns the agent can query |
| `/good [note]` | Rate the last answer as good |
| `/bad [note]` | Rate the last answer as bad |
| `/help` | Show the command list |

Anything else is treated as a question.

## Demo script

Start `uv run ria chat --user alice` and type these in order:

1. `What data do we have and what can I ask about?`
   Schema answer from the cached schema; no BigQuery query.
2. `Compare revenue of Levi's and Carhartt over the last 12 months.`
   Analysis. The SQL is filtered to Alice's brands by the guard.
3. `Why is the gap so large?`
   Follow-up that uses the conversation so far.
4. `How did Calvin Klein sell last month?`
   Calvin Klein is not one of Alice's brands, so the answer is the scoped no-data message.
5. `Save this as a report.`
   Saves the last analysis and queues it as a Golden Trio candidate.
6. `Use tables from now on.`
   Stores a preference; later analyses append the result rows as a table.
7. `Delete the reports from this conversation.`
   Lists the matching reports and asks for confirmation. Type `yes` to delete. Anything
   else cancels.

After each answer the CLI prints `[turn <turn_id> · <outcome>]`.

## Inspecting what happened

```bash
uv run ria trace <turn_id>      # nodes, model calls, SQL attempts, guard events for one turn
uv run ria stats                # success, refusal, error and fallback rates, outage_turns, repairs, latency, ratings, correction rate
uv run ria stats --since 2026-10-16T00:00:00
```

## Golden Trios

```bash
uv run ria golden list                    # all Trios with status and source
uv run ria golden list --status pending   # candidates from saved reports
uv run ria golden promote <id>
uv run ria golden reject <id>
uv run ria init                           # embeds newly added Trios
```

Only promoted Trios are retrieved as examples for SQL generation.

## Evaluation

```bash
uv run ria eval                  # all 24 cases in evals/questions.yaml, live services
uv run ria eval --only pii-request
```

Each case gets deterministic checks (intent, outcome, repairs, tables in the SQL, brand
scope, no PII patterns in the answer) and an LLM judge score from 1 to 5. Results are
written to `evals/results-<timestamp>.json`. The command exits with 1 if any deterministic
check fails.

The run makes roughly 100 model calls on one key, so on the free tier expect 429 fallbacks
and primary-down windows. Use `--only <case>` to run cases one at a time.

## Changing the persona and the definitions

Edit `data/persona.md` while a chat is running. The file is read on every turn, so the next
answer uses the new tone and instructions. No restart is needed.

Org defaults live in `data/org_defaults.yaml` (churn window, revenue statuses, minimum
group size). The data or admin team owns the file, and it is read on every turn. It feeds
both the SQL guard and the definitions the model sees. User preferences (format, depth,
charts) are presentation only and never change a definition. A window or statuses named in
a question override the defaults for that answer only, and the answer says so.

### Persona changes

```bash
uv run ria persona check my-persona.md            # lint only, changes nothing
uv run ria persona check my-persona.md --live     # also runs the 24 eval cases with it
uv run ria persona publish my-persona.md [--live] # same checks, then replaces data/persona.md
```

The lint rejects an empty persona, more than 4000 characters, and any line that tries to
override instructions, touches scope, brands, email, SQL, SELECT, delete, PII, the minimum
group size, the output language (English only), or redefines churn or revenue; naming a
metric ("round revenue to the nearest thousand") is fine. A report header may not name a
brand, because the persona is shared by every user and scope is per user. `--live` requires zero deterministic eval failures and a judge mean of at least 4.0.
Publish saves the current persona to `data/persona_versions/<UTC timestamp>.md` first, so
rolling back is publishing a saved version. There is no manual approval step.

## Tests

```bash
uv run pytest
```

259 tests, no network or credentials needed. Model, embedding and MCP calls are replaced by
scripted fakes.

## Free-tier note

Google does not publish exact free-tier limits for the Gemini API, and they are low for
Flash models. When a call returns 429 or a 5xx error, the agent retries up to 3 times on
the primary model (waiting 2 s, then 4 s), then switches to `gemini-3.5-flash-lite`. After
3 failed attempts on the primary it skips the primary for 60 s. An answer that used the
backup model ends with "Note: answered with the backup model." If both models fail, the
turn ends with "The language model is unavailable right now" and the conversation
continues. An analysis turn makes 3 to 5 model calls and 1 embedding call, so a few
questions per minute is the practical rate on the free tier.

## Client answers

OpsFleet answered the clarification questions after the first version. Details are in
[technical-design.md section 13](docs/technical-design.md#13-client-answers-and-open-items).

| Question | Answer | What it changed |
|---|---|---|
| Which products does a manager see? | Their brands; the CEO sees all | Scope key is `products.brand`; demo users own brands |
| Customer-level questions for a manager? | Answerable, filtered to their data | Users-only and orders-only queries are scoped by a semi-join instead of refused |
| Churn definition? | No purchase in 90 days, overridable | `data/org_defaults.yaml`, read every turn; answers state the definition used |
| Revenue definition? | Complete, Shipped, Processing | Same file |
| Demographics? | Aggregates only, never per person | Guard rejects ids next to demographics; groups need at least 10 customers |
| Delete and sharing? | Hard delete; reports are private to the author | No change |
| Learning signals? | All five | `/good`, `/bad` ratings and correction tracking on traces |
| Degraded mode? | Ours to design | Slash commands work with no language model |
| Distribution centres? | Brand is the only access dimension | No change |
| Do responsibilities change? | No, they are static in the users file | No change |
| Other languages? | English only | The persona cannot change the output language |
| Timings and thresholds? | Ours to pick and justify | Unchanged; justified in the technical design |
| Disclose the backup model? | Our call | Disclosed: a note after the answer |
| Who needs to know about an outage? | Users and the ops team | User message plus `outage_turns` in `ria stats`; ops alert defined in the HLD |
| Org-wide defaults? | A config file owned by the data or admin team, user preferences on top | `data/org_defaults.yaml` feeds the guard and the prompts |
| Persona changes and QA? | Must pass an automated gate, no manual approval | `ria persona check` and `ria persona publish` |

Open items: who approves Golden Trio candidates, and who scores the gold set (the evaluation
questions with expected answers).

## Repository layout

```
data/
  golden_trios/*.yaml   15 seed Trios (question, SQL, analyst report, tags)
  persona.md            tone and instructions, read every turn
  persona_versions/     previous personas saved by `ria persona publish` (gitignored)
  org_defaults.yaml     churn window, revenue statuses, minimum group size
  users.yaml            demo users and their brands
docs/
  hld.md                production architecture and diagrams
  technical-design.md   per-requirement design and error matrix
evals/
  questions.yaml        24 evaluation cases
src/ria/
  cli.py                Typer commands: init, doctor, chat, trace, stats, golden, persona, eval
  bootstrap.py          init logic and the chat loop, testable without a terminal
  config.py             settings from env: models, budgets, caps
  db.py                 SQLite schema and connection with sqlite-vec loaded
  tools.py              MCP client toolbox (stdio subprocess) and a fake for tests
  graph/                LangGraph state, nodes, prompts, and build_graph
    commands.py         slash-command parsing and the /help text
    prerouter.py        pre_router node: commands skip the LLM router
  guards/               SQL guard, PII masker, schema digest
  llm/                  Gemini client with retry and fallback; scripted fake
  golden/               Trio loader, embedder, sqlite-vec store, retrieval
  memory/prefs.py       preference extraction and storage
  persona/loader.py     persona file loader
  persona/gate.py       persona lint, live eval gate, versioned publish
  mcp_server/           FastMCP server, BigQuery runner with circuit breaker, report store
  observability/        trace store, stats, trace renderer
  evals/                eval checks and runner
tests/                  pytest suite, offline
```

## Prototype vs design only

| Area | In the prototype | Design only (see docs) |
|---|---|---|
| Safety and PII | SQL guard, brand scope rewrite, demographics rules, output masker, off-topic refusal | Authorized views, DLP API pass, least-privilege service account |
| Report deletion | Interrupt-based confirmation that survives a restart; hard delete | Confirmation button in a chat UI, audit log |
| Resilience | Retry and fallback with a backup-model note, call budget, repair loop, byte cap, timeout, circuit breaker, slash commands with no model, `outage_turns` | Provisioned throughput, regional failover, ops alert |
| Observability | Per-turn traces, `ria trace`, `ria stats`, optional LangSmith | Cloud Logging and Trace, alerts |
| Golden Bucket | Seed Trios, sqlite-vec retrieval, candidate promotion commands | GCS bucket, Vertex AI Vector Search, weekly review job |
| Learning | Regex preference memory, Trio candidates from saved reports, `/good` and `/bad` ratings, correction tracking | Eval growth from traced failures, prompt-regression runs |
| Evaluation | 24-case eval with deterministic checks and an LLM judge | CI gate, rating buttons in a chat UI, executive review sessions |
| Persona and org defaults | `data/persona.md` and `data/org_defaults.yaml` hot reload; persona gate with lint and live eval, versions under `data/persona_versions/` | Versioned GCS objects with an admin page that runs the same gate |
| Not built | | Web UI, real identity, charts, email delivery, web search |

## Example run

Transcript to be captured after `ria init` on a machine with credentials; see the demo script above.
