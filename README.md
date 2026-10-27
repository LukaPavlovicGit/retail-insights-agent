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
- [docs/hld.md](docs/hld.md): production architecture on Google Cloud, the flow of one
  analysis turn, the delete confirmation flow, component choices, and non-functional notes.
- [docs/technical-design.md](docs/technical-design.md): one section per assignment
  requirement with the module and test that implement it, the error handling matrix, and
  the client's answers with what each one changed.
- [docs/data-model.md](docs/data-model.md): the warehouse tables and their relationships, the agent's own store, and how each table is used.

Why LangGraph, and my experience with it: [technical-design.md section 9](docs/technical-design.md#9-why-these-choices)

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

`uv sync` installs the pinned versions from `uv.lock`; there is no requirements.txt.

- `ria doctor` checks the Gemini key, Google credentials, a BigQuery dry run, and one Gemini
  call.
- `ria init` loads the demo users and the 15 seed Golden Trios into `ria.db`, embeds the
  Trio questions, caches the live BigQuery schema in `data/schema.json`, and checks that the
  brands in `data/users.yaml` exist in the dataset. Run it again after promoting a Trio.
  Conversation state (LangGraph checkpoints) lives in a second file, `ria-checkpoints.db`.
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
   Calvin Klein is not one of Alice's brands. The guard keeps the query inside her scope,
   the query returns no rows, and the turn ends with the no-data message that names her
   brands and shows the SQL. No repair is attempted on an empty result.
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

The checks are an offline lint and, with `--live`, the eval set (zero deterministic failures,
judge mean at least 4.0). Publish saves the current persona to
`data/persona_versions/<UTC timestamp>.md` first, so rolling back is publishing a saved
version. There is no manual approval step. The lint rules are in
[technical-design.md section 6](docs/technical-design.md#6-quality-assurance), and what a
persona may not change is in [section 8](docs/technical-design.md#8-persona-management).

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
| Customer-level questions for a manager? | Answerable, filtered to their data | Users-only and orders-only queries are scoped by a semi-join |
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
  assignment.md         the assignment brief
  hld.md                production architecture and diagrams
  technical-design.md   per-requirement design and error matrix
  data-model.md         warehouse tables and the agent's own store
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

The demo script above, captured on 2026-10-26 with `gemini-3.5-flash-lite` as the primary
model (the default primary was out of free-tier quota that day). Long answers are shortened
where marked; everything else is verbatim.

```
Chatting as Alice Chen (VP Denim and Workwear). Thread demo-readme. Ctrl-D to exit. Type /help for commands.

you> What data do we have and what can I ask about?
We have the `bigquery-public-data.thelook_ecommerce` dataset, which contains five interconnected tables covering orders, order items, products, and users.

The `orders` table tracks the lifecycle of customer orders (such as status, dates, and item counts), while the `order_items` table details every individual item sold, including its sale price. The `products` table acts as our catalog, containing costs and retail prices that allow us to calculate margins, along with categories, brands, and departments. The `users` table provides customer demographics like age, location, and traffic source, though personal customer details such as names, email addresses, and specific coordinates are not available.

These tables connect through shared identifiers: `order_items` joins to `orders` via `order_id`, to `products` via `product_id`, and to `users` via `user_id`.

You can ask questions such as:
1. Which product categories generated the highest revenue and profit margins last quarter?
2. What are the top traffic sources for our most valuable customer demographics by state?
3. How do return rates vary across different product brands and departments?
[turn f4b13606b84d · ok]

you> Compare revenue of Levi's and Carhartt over the last 12 months.
Carhartt outperforms Levi's in both total sales and margin rate over the last 12 months, where revenue is defined as the sum of order_items.sale_price for items with status Complete, Shipped, or Processing.

Carhartt generated $56,000 in revenue across 818 units sold, achieving a strong margin rate of 53.5%.

Levi's brought in $25,000 in revenue across 473 units sold, with a slightly lower margin rate of 50.0%.

Carhartt's higher volume and better margin make it the primary revenue driver between the two brands in your portfolio.

Action: Merchandising: Review Levi's pricing and promotional terms to close the margin and volume gap with Carhartt.
[turn ac34ce104943 · ok]

you> Why is the gap so large?
Carhartt's revenue lead over Levi's is driven primarily by dominance in higher-priced Outerwear & Coats, where revenue is defined as the sum of `order_items.sale_price` for completed, shipped, or processing items over the last 12 months.

Carhartt sold 276 units of Outerwear & Coats at an average price of $124.13, generating $34,000 in revenue alone—more than Levi's total brand revenue.

Levi's volume is concentrated in lower-priced Jeans, selling 298 units at an average price of $47.29 for $14,000 in revenue.
[... two more paragraphs on secondary categories and margins ...]
Action: Merchandising: Expand the high-end outerwear assortment for Levi's to capture higher average sale prices.
Action: Merchandising: Investigate the drivers behind Levi's low unit volume in categories outside of core jeans.
[turn 9d10cb24ff94 · ok]

you> How did Calvin Klein sell last month?
Calvin Klein is outside your scope, as you only manage Levi's, Carhartt, Dockers, and Wrangler.

The available brands generated $241,000 in total revenue across all categories, where revenue is defined as the sum of order_items.sale_price for items with status Complete, Shipped, or Processing.
[... per-brand paragraphs for Carhartt, Levi's, Wrangler and Dockers ...]
[turn 91bd7d68ccec · ok]

you> Save this as a report.
Saved report #3: Monthly Brand Revenue and Sales Performance. It is also queued as a Golden Trio candidate for analyst review (`ria golden list`).
[turn cd2865c24e56 · saved]

you> Use tables from now on.
Noted. From now on: Format: table. Depth: standard. Charts: no.
[turn e171b02f147f · ok]

you> Delete the reports from this conversation.
This will permanently delete 1 report(s):

- #3  Monthly Brand Revenue and Sales Performance  (2026-10-26T15:51:27+00:00)

Type 'yes' to confirm, anything else to cancel.

confirm> yes
Deleted 1 report: Monthly Brand Revenue and Sales Performance.
[turn 7afbb6503770 · deleted]
```

Two things to notice. The schema answer says "five" tables where there are four: a model slip
on a cached-schema answer with no query behind it. The Calvin Klein turn shows why the
"repair once on an empty result" rule was removed the next day: the first query filtered on
Calvin Klein inside Alice's scope and returned nothing, and the repair widened it to all of
her brands and dropped the month, so the answer is about a different question than the one
asked. The eval case `scoped-outside` expects `no_data` for exactly this question. Since
then an empty result ends the turn at once with the scoped no-data message and the SQL that
ran; step 4 above now produces that message.
