# Technical Design: Retail Insights Agent

This document covers the eight assignment requirements, one section each, in the same
shape: what the requirement asks, how the prototype does it (module and the test that
proves it), and how production extends it. Cross-cutting sections follow. Architecture
diagrams are in [hld.md](hld.md).

Requirements 2, 3, 5 and 7 are fully implemented in the prototype. Requirements 1, 4, 6 and
8 have a thin working version in the prototype plus a production design.

All paths below are relative to `src/ria/` unless they start with `tests/`, `data/` or
`evals/`. The test suite has 204 tests and runs offline (`uv run pytest`).

OpsFleet answered the clarification questions after the first version. Section 13 lists the
answers, what each one changed, and the items still open.

## Contents

1. [Hybrid intelligence (Golden Bucket)](#1-hybrid-intelligence-golden-bucket)
2. [Safety and PII](#2-safety-and-pii)
3. [High-stakes oversight](#3-high-stakes-oversight)
4. [Continuous improvement](#4-continuous-improvement)
5. [Resilience](#5-resilience)
6. [Quality assurance](#6-quality-assurance)
7. [Observability](#7-observability)
8. [Persona management](#8-persona-management)
9. [Why these choices](#9-why-these-choices)
10. [Data flow](#10-data-flow)
11. [Error handling and fallback matrix](#11-error-handling-and-fallback-matrix)
12. [Setup and example run](#12-setup-and-example-run)
13. [Client answers and open items](#13-client-answers-and-open-items)

---

## 1. Hybrid intelligence (Golden Bucket)

**Asks.** Combine the model with a curated set of expert examples (question, SQL, report),
use them to improve answers, and grow the set over time.

**Prototype.**

- **Trio format.** One YAML file per Trio in `data/golden_trios/`, with `question`, `sql`,
  `report`, `tags`, `author`, `created_at`. `golden/loader.py` (`load_trios_dir`) rejects a
  file with a missing `question`, `sql` or `report` and names the file in the error.
- **Seeding.** 15 seed Trios cover the assignment's example questions: top customers, monthly
  revenue, category revenue by quarter, comparing two products, a churn proxy, a state
  comparison, a quarterly report, return rate by category, new vs returning customers,
  traffic source, top brands in a category, revenue month to date, order status funnel,
  delivery time, and a schema overview. All 15 are tagged `all`. The top-customers Trio (01)
  returns no demographic columns next to customer ids, and the churn Trio (05) uses the
  90-day definition. `ria init` (`bootstrap.run_init`) replaces the seed rows, keeps user-sourced Trios, and
  embeds every Trio that has no vector yet.
- **Embeddings.** `golden/embed.py` (`GeminiEmbedder`) calls `gemini-embedding-001` with
  `output_dimensionality=768`, task type `RETRIEVAL_DOCUMENT` for Trio questions and
  `RETRIEVAL_QUERY` for the user's question.
- **Index and retrieval.** Vectors live in a sqlite-vec `vec0` table `trio_vec` with
  `distance_metric=cosine` (`db.py`). `golden/store.py` (`TrioStore.search`) runs KNN, reads
  `max(4k, 8)` neighbours, then keeps up to k=3 that pass three filters: cosine distance at
  most 0.5, `status = 'promoted'`, and tags that contain `all` or share a value with the
  user's brands (an unrestricted user sees every tag).
- **Use in prompts.** `graph/analysis.py`: `generate_sql` puts the retrieved Trios into the
  SQL prompt as worked examples with question, SQL and analyst report ("adapt their logic; do
  not copy blindly"). `analyze` puts the same Trios in without the SQL, as a style reference
  for the written answer.
- **Business definitions.** `data/definitions.md` holds the agreed definitions: churn is no
  purchase in the last 90 days; revenue is the sum of `order_items.sale_price` for status
  Complete, Shipped or Processing (Cancelled and Returned excluded). `Deps.definitions()`
  (`persona/loader.py`, `load_definitions`) reads the file on every call, like the persona,
  and falls back to built-in text if it is missing or empty. The text goes into the SQL,
  repair and analysis prompts. The analysis prompt tells the model to state the definition
  used whenever it reports churn or revenue, and to use the user's own window or statuses
  for that answer (and say so) when the question names them.
- **Update loop.** Saying "save this as a report" saves the report (`graph/library.py`,
  `save_report`) and inserts a `pending` Trio with `source='user'`, tagged with the user's
  brands (`all` for an unrestricted user). `ria golden list`, `ria golden promote <id>` and `ria golden reject <id>`
  (`cli.py`) are the review step. The next `ria init` embeds the new Trio; only promoted
  Trios are ever retrieved.
- **Guard compatibility.** Every seed Trio's SQL passes the SQL guard for the CEO and for a
  scoped brand user, including the demographics rules in section 2.
- **Tests.** `tests/test_golden.py`: `test_search_filters_by_status_and_tags`,
  `test_search_respects_k_and_distance`, `test_replace_seed_keeps_user_trios`,
  `test_load_trios_dir_reports_bad_file`, `test_seed_trios_pass_guard_for_scoped_and_unscoped_users`.
  `tests/test_graph_analysis.py`: `test_generate_sql_strips_fences_and_uses_examples`,
  `test_prompts_carry_business_definitions`. `tests/test_memory_persona.py`:
  `test_load_definitions`.
  `tests/test_graph_library.py`: `test_save_report_saves_and_creates_candidate_trio`.

**Production.**

- Trio documents live in a GCS bucket, one object per Trio, with metadata in Cloud SQL.
- Embeddings live in a Vertex AI Vector Search index with brand tags as restricts, so the
  scope filter runs inside the index query.
- A weekly Cloud Scheduler job collects candidates and applies four gates before a human sees
  them: the query executed without error, the answer was not rated `/bad` or marked
  corrected (section 4), a PII scan
  (DLP API) of the SQL and report is clean, and the SQL passes the guard for every role.
  An analyst then approves or rejects each candidate on a review page.
- Each approved batch builds a new index version. The agent switches to it by changing one
  config value, so a bad batch is rolled back by pointing at the previous version.

## 2. Safety and PII

**Asks.** Keep personal data out of answers, refuse malicious requests, and limit each user
to the products related to them. OpsFleet confirmed that "related products" means brands,
that the CEO sees everything, and that demographics may appear only in aggregates.

**Prototype.** Four layers. The first is in the model path; the other three are plain Python
that runs no matter what the model produces.

| Layer | Module | What it does |
|---|---|---|
| 1. Intent gate | `graph/router.py` | One structured-output call classifies the message into six intents. `off_topic` covers prompt-injection attempts, requests for the system prompt, and unrelated tasks; it returns a fixed refusal with no further calls. An empty message is `off_topic` without a model call. Slash commands are matched in code before the router (section 5) and never reach the model. |
| 2. SQL guard | `guards/sql_guard.py` | Parses and rewrites the generated SQL before it reaches BigQuery (details below). Only the four schema tables (orders, order_items, products, users) are allowed, so other tables in the dataset, such as events, are rejected. Applies the demographics rules for every user, the CEO included. |
| 3. Scope injection | `guards/sql_guard.py` | For scoped users, adds a brand filter to every SELECT that reads products or order_items, and a semi-join through order_items and products to every SELECT that reads only orders or users. |
| 4. Output masker | `guards/pii_mask.py`, called from `graph/lifecycle.py` `finalize` | Regex pass over every rendered answer on every graph path, including errors and refusals. The delete-confirmation list the CLI prints (report titles generated by the model at save time) comes from the interrupt payload, outside `finalize`, so it is not masked. |

PII is also kept out of the model's view: `guards/schema.py` (`schema_digest`) omits the
personal columns from the schema text in every prompt and states that they do not exist.

**PII columns** (on `users`): `email`, `first_name`, `last_name`, `street_address`,
`postal_code`, `latitude`, `longitude`. Demographic columns (`DEMOGRAPHIC_COLUMNS` in
`guards/schema.py`: `state`, `city`, `country`, `age`, `gender`, `traffic_source`) may be
queried, but only in aggregates (step 5). `created_at` is not a demographic column.

**Scope key.** The scope is a column on `products`, set by `Settings.scope_column` (default
`brand`, env `RIA_SCOPE_COLUMN`). Each user's values come from the `brands` key in
`data/users.yaml`; `["*"]` (the CEO) means unrestricted. `ria init` reads
`SELECT DISTINCT brand` from `products` and warns about any value in `users.yaml` that is not
in the dataset.

**SQL guard steps** (`guard_sql`, sqlglot BigQuery dialect). Each step appends an event such
as `pii:ok` or `demographics:rejected` to the turn's `guard_events`.

1. Parse. Reject a parse error, more than one statement, or anything that is not a SELECT or
   a set operation (UNION etc.).
2. Dataset. Reject any table outside `bigquery-public-data.thelook_ecommerce`. CTE references
   are allowed.
3. Resolve. Run sqlglot's `qualify` with the schema so aliases and `*` resolve to real
   columns. An unknown column is rejected with the resolver's message, which the repair loop
   feeds back to the model.
4. PII. Reject any column that resolves to a PII column on `users`, in direct, aliased, CTE,
   or `SELECT *` form (star expansion produces the PII columns). Reject whole-row references
   to `users` such as `SELECT u`, `TO_JSON_STRING(u)` or `STRUCT(u)`.
5. Demographics (every user, the CEO included). A customer id is `users.id`,
   `order_items.user_id` or `orders.user_id`.
   - Rule A, no individuals with demographics. The check covers the whole query, not one
     SELECT at a time. If any select list anywhere in the query (outer query, CTE or
     subquery) returns a customer id, and the query references a demographic column anywhere
     (select list, WHERE, GROUP BY, any SELECT), the query is rejected with code
     `demographics`. A customer id inside `COUNT(...)` or `APPROX_COUNT_DISTINCT(...)` is not
     exposure; inside any other aggregate (for example `MAX` or `ARRAY_AGG`) it is. So
     "top 20 customers with their city" is rejected, and "customers per state" is allowed.
     Checking the whole query closes the bypass where a CTE pairs ids with demographics and
     the outer query only selects ids.
   - Rule B, minimum group size. Each SELECT that groups by a demographic column (directly or
     by position, `GROUP BY 1`) and reads `users`, `order_items` or `orders` directly gets
     `HAVING COUNT(DISTINCT <customer id>) >= 10`, ANDed into any existing HAVING. The id is
     `users.id` when users is a source, else `order_items.user_id`, else `orders.user_id`.
     Event `demographics:min_group`. The threshold is `Settings.min_group_size` (default 10,
     env `RIA_MIN_GROUP_SIZE`). Groups below 10 customers are dropped from the result, not
     reported as zero.
6. Scope (scoped users only). For each SELECT in the tree, look at the dataset tables in its
   own FROM and JOIN clauses:
   - `products` present: add `<alias>.brand IN (...)` to that SELECT's WHERE.
   - `order_items` present but not `products`: add
     `INNER JOIN products AS ria_p ON ria_p.id = <order_items alias>.product_id`, then add
     `ria_p.brand IN (...)`. Event `scope:join_added`.
   - Only `orders` and/or `users`: add a semi-join predicate. With `orders` present:
     `<orders alias>.order_id IN (SELECT oi.order_id FROM order_items oi JOIN products p ON p.id = oi.product_id WHERE p.brand IN (...))`.
     Otherwise, with `users`: `<users alias>.id IN (SELECT oi.user_id FROM ... WHERE p.brand IN (...))`.
     Event `scope:semijoin`. The grain of the outer query is kept (no row multiplication),
     so "signups per month" for a scoped user counts customers who bought one of their
     brands. Before the client answers this case was rejected; it is now answered.
   - No dataset tables (the SELECT reads only a CTE or subquery): skip; the inner SELECT is
     filtered instead.
   Every SELECT that reads a dataset table is filtered on its own, so an outer query over
   `orders` with `products` only inside a subquery gets the semi-join as well as the inner
   predicate. If no SELECT received a filter (the query reads no dataset table), the query
   is rejected with code `scope`.
7. LIMIT. Add `LIMIT 200` if absent; clamp a larger limit to 200.
8. Dry run. The guard node (`graph/analysis.py`, `guard_sql`) calls MCP `bq_dry_run` on the
   rewritten SQL and rejects the query if the estimate exceeds 1 GB.

Scope values are passed to sqlglot as literals, so a brand with a quote in it (`Levi's`) is
escaped (`test_scope_value_with_quote_is_escaped`).

**Output masker** (`mask_pii`): emails; phone numbers with separators, parentheses around
the area code, or a leading `+` (bare 10-digit runs are order and user ids and are left
alone); street addresses of the form number, capitalised words, street type, where the
street type must end at punctuation, a newline, a table pipe, end of text, or be followed by
an Apt/Suite/Unit/# suffix. The count of masked items is computed but not yet recorded in
the trace.

`save_report` also runs the masker, so saved report bodies and pending Trio candidates are masked before storage.

**Tests.** `tests/test_sql_guard.py` (48 collected tests), including
`test_rejects_pii_aliased`, `test_rejects_pii_star_on_users`, `test_rejects_pii_inside_cte`,
`test_rejects_whole_row_users_reference`, `test_scope_adds_join_when_only_order_items`,
`test_scope_semijoin_for_users_only`, `test_scope_semijoin_for_orders_only_keeps_existing_where`,
`test_scope_orders_with_products_only_in_subquery_gets_semijoin_and_inner_predicate`,
`test_scope_column_setting`, `test_scope_injected_inside_cte`,
`test_demographics_rejects_individual_with_demographic`, `test_demographics_rejects_bypass`,
`test_demographics_allows_count_distinct_with_demographic`,
`test_demographics_min_group_added_on_group_by`,
`test_demographics_min_group_ands_into_existing_having`,
`test_demographics_min_group_size_setting`, `test_limit_added_and_clamped`.
`tests/test_pii_mask.py` (7 tests), including `test_leaves_ids_and_counts_alone`.
`tests/test_graph_integration.py`: `test_off_topic_is_refused_with_one_call`.

**Production.**

- BigQuery authorized views per brand set, so scope is enforced by IAM as well as by the
  rewrite. The guard stays as the first line because it gives the model a usable error.
- The tool service runs as a service account with only `bigquery.jobs.create` and
  `bigquery.tables.getData` on the dataset or views.
- A Cloud DLP API inspection pass on the final answer and on saved reports, after the regex
  masker.
- A prompt-injection eval set (instructions hidden in questions, in report titles, and in
  retrieved Trios) run in CI next to the main evals.

## 3. High-stakes oversight

**Asks.** Deleting reports must require explicit human confirmation.

**Prototype.** The flow is `resolve_targets -> confirm_delete (interrupt) -> resume`, in
`graph/library.py`. The state diagram is in [hld.md section 4](hld.md#4-report-deletion-with-confirmation).

1. The router returns `report_delete` with `report_filter_scope` (`conversation`, `text` or
   `all`) and, for `text`, `report_filter_text`. The `/delete <filter>` command sets the same
   fields in code (`pre_router`), with no model call.
2. `resolve_targets` calls MCP `reports_search` with the current `user_id` and the filter.
   The candidate list is filtered by `user_id` in the SQL of `ReportStore.search`
   (`mcp_server/reports.py`), so the list can only contain the user's own reports. No model
   sees or chooses the list. If the filter is unclear, the agent asks which reports; if
   nothing matches, it says so and the turn ends.
3. `confirm_delete` calls `interrupt({"action": "delete_reports", "reports": [{"id", "title", "created_at"}, ...]})`.
   LangGraph saves the thread state in the checkpointer and the turn returns with outcome
   `needs_confirmation`.
4. The CLI (`bootstrap.chat_loop`) prints the list with ids and titles and asks for
   confirmation. Only `yes` or `y` (trimmed, any case) confirms (`parse_confirmation`). Any
   other answer, an empty line, Ctrl-C or Ctrl-D cancels.
5. The CLI resumes with `Command(resume={"confirmed": bool})`. On `True` the node calls MCP
   `reports_delete(user_id, ids)`, which also filters by `user_id` in its SQL, and reports
   the count and titles. On anything else it replies "Cancelled. Nothing was deleted."

Nothing in `confirm_delete` before `interrupt()` has side effects, so LangGraph re-running
the node on resume is safe. Because the pending deletion is in the checkpoint, it survives a
process restart: the test closes the checkpointer, opens a new one on the same file, and
resumes.

**Tests.** `tests/test_graph_integration.py`:
`test_save_then_delete_with_confirmation_survives_restart`. `tests/test_graph_library.py`:
`test_delete_interrupts_then_deletes_on_yes`, `test_delete_cancelled_on_no`,
`test_resolve_targets_by_scope`. `tests/test_cli.py`: `test_parse_confirmation`,
`test_chat_loop_handles_confirmation_and_eof`. `tests/test_reports.py`:
`test_save_search_delete_scoped_to_user`.

**Production.** The same interrupt drives a confirmation card in the chat UI with Delete and
Cancel buttons; the button sends the resume command for that thread. Every deletion writes an
audit row (who, when, which report ids and titles, thread id). Deletion stays a hard delete:
OpsFleet confirmed that the confirmation is the safeguard and asked for no recovery window.

## 4. Continuous improvement

**Asks.** Learn from use at the user level (preferences) and at the system level.

**Prototype, user level.** `memory/prefs.py`.

- Preferences: `format` (table, bullets, prose), `depth` (brief, standard, deep), `charts`
  (on/off), stored per user in `user_prefs`.
- Extraction is regex based, with no model call (`extract_preferences`). It looks for explicit
  statements: "use tables", "bullet points", "in prose", "more detail", "shorter", "no
  charts". A chart preference needs a cue word (include, add, show, want, with, prefer, use,
  give, make, like) before the chart noun, and "chart of" / "graph of" is not a preference.
  "Why did the graph of sales dip?" changes nothing; "I'd like a graph with the answer" turns
  charts on.
- Brevity is a three-strike rule. Each "shorter" / "too long" request increments
  `brevity_streak`; on the third, `depth` drops one step and the streak resets. A request for
  more detail raises depth one step at once and resets the streak.
- Two places apply preferences. A message that is only a preference is routed to
  `ack_preference` (`graph/router.py`), which applies and confirms it. On every other turn,
  `update_memory` (`graph/lifecycle.py`) runs the same extraction on the user's message, so
  "use a table and show revenue by month" both answers and updates the preference.
- Preferences are used in two ways. A one-line summary goes into the router, analysis and
  schema prompts, with depth and format instructions in the analysis prompt. `render`
  (`graph/analysis.py`) applies them deterministically: a Markdown table of the rows when
  format is `table`, and the SQL when depth is `deep`. The `charts` flag is stored and shown
  but nothing renders charts yet (see the extension points in the HLD).

**Prototype, system level.** OpsFleet approved all five learning signals. As built:

| Signal | How it is captured | Where it is stored | Where it shows up |
|---|---|---|---|
| Stated preference | Regex extraction, above | `user_prefs` | Every prompt and the renderer |
| Repeated request | Three "shorter" requests lower `depth` one step | `user_prefs.brevity_streak` | Same |
| Saved report | `save_report` inserts a `pending` Trio (section 1) | `trios` | `ria golden list`; retrieved once promoted |
| Rating | `/good [note]` or `/bad [note]`, recognised by the pre-router with no model call; the `rating` node writes +1 or -1 and the note | `traces.rating`, `traces.rating_note` on the previous answer's trace | `ria trace` (`rating: +1 (note)`), `ria stats` (`rated good/bad`) |
| Correction | The router's structured output has `is_correction` ("the user is correcting or re-asking because the previous answer was wrong or not what they meant"); when true the router sets `corrected = 1` on the previous answer's trace and still routes the message normally | `traces.corrected` | `ria trace` (`corrected: yes`), `ria stats` (`correction_rate` = corrected analysis turns / analysis turns) |

"The previous answer" is `last_turn_id`, conversation-level state that `finalize` sets on
every turn whose outcome is an answer (`ok`, `no_data`, `gave_up`, `refused`, `error`,
`saved`, `deleted`, `cancelled`). Ratings, `/reports`, `/report <id>` and `/help` do not move
it, so `/good` after `/reports` still rates the last real answer. A `/good` with no earlier
answer in the thread replies "Nothing to rate yet" and records nothing.

Ratings and corrections are recorded and reported; nothing acts on them automatically yet.
The production items below use them.

**Tests.** `tests/test_memory_persona.py`: `test_three_brevity_requests_lower_depth`,
`test_depth_up_resets_streak`, `test_chart_mentions_in_questions_are_not_preferences`,
`test_extract_format_preferences`. `tests/test_graph_integration.py`:
`test_preference_then_table_render`, `test_rating_applies_to_last_answer_not_to_listing`,
`test_good_as_first_turn_does_not_become_the_rating_target`,
`test_correction_marks_previous_turn`. `tests/test_graph_foundation.py`: `test_rating_node`,
`test_router_marks_previous_turn_corrected`. `tests/test_trace.py`:
`test_set_rating_and_corrected_round_trip`, `test_stats_ratings_and_correction_rate`.
`tests/test_graph_analysis.py`: `test_render_table_and_deep`.

**Production.**

- Preferences keyed by the identity provider's user id, in Cloud SQL.
- A small model call replaces the regex when the regex finds nothing but the router said
  `preference_update`, with the result shown back to the user for confirmation.
- Eval growth from traced failures: every turn with outcome `gave_up` or `error`, rated
  `/bad`, or marked corrected becomes a candidate eval case. An analyst writes the expected intent, tables
  and outcome, and the case joins `evals/questions.yaml`.
- A weekly prompt-regression run executes the full eval set against the current prompts,
  persona and Trio index version and compares with the previous week.

## 5. Resilience

**Asks.** Handle model, warehouse and query failures gracefully, and control cost.

**Prototype.**

- **LLM wrapper** (`llm/gemini.py`, `GeminiClient`). Each call tries the primary model
  (`gemini-3.8-flash`) up to 3 times. On a retryable error (429, 500, 503, 504, or messages
  such as `RESOURCE_EXHAUSTED`, `UNAVAILABLE`, timeouts) it waits 2 s after the first failure
  and 4 s after the second. After the third, it moves to the fallback model
  (`gemini-3.5-flash-lite`) with the same 3-attempt policy. A non-retryable error (for
  example a 400) skips the remaining attempts and goes straight to the fallback. A
  structured-output parse failure is retried on the same model without waiting, then falls
  back. The primary is marked down for 60 s only when all 3 attempts failed with retryable
  errors; during that window calls go directly to the fallback. A 400 or parse failures do
  not mark it down. If both models fail, `LLMUnavailable` is raised. The underlying LangChain
  client has its own retries disabled (`max_retries=0`) and a 60 s timeout.
- **Graph error conversion.** `graph/nodes_common.py` (`timed`) wraps every node and turns
  `LLMUnavailable` and `BudgetExceeded` into an `error` field. The graph routes to `fail`,
  which tells the user the model is unavailable or the per-question budget was reached, and
  the turn still writes its trace.
- **Per-turn budget.** `call_llm` refuses a ninth model call in one turn (8 calls maximum).
  A normal analysis turn uses 3 to 5.
- **SQL repair loop.** Guard rejections, dry-run errors and query errors go to `repair_sql`
  with the full attempt history (each attempt's SQL and error or row count). Budget: 2
  repairs, so at most 3 SQL attempts. Each repaired query goes through the guard again. On
  exhaustion, `give_up` lists what was tried and suggests rephrasing.
- **Embedding failure.** The `retrieve_trios` node catches any exception from the embedding
  call, continues the turn without examples, and records `retrieval:failed:<type>` in the
  trace's guard events.
- **Empty results.** The first empty result triggers one repair (the model often filters too
  strictly; the repair prompt says to loosen filters). A second empty result ends the turn
  with outcome `no_data`, a message that names the user's brands, and the last SQL.
- **BigQuery limits** (`mcp_server/bigquery.py`, `BigQueryRunner`). Dry run before every
  query; reject above 1 GB (`RIA_BQ_MAX_BYTES`); `maximum_bytes_billed` set on the real
  query as a second cap; 60 s timeout; at most 200 rows fetched.
- **Error classes.** `classify_error` maps exceptions to `timeout`, `bytes_cap`, `quota`,
  `syntax`, `unknown`; an open breaker returns `circuit_open`. `bytes_cap` and
  `circuit_open` end the turn with a specific message and no repair; the others go to repair.
- **Circuit breaker.** 3 consecutive failures open it for 5 minutes. `syntax` and
  `bytes_cap` do not count, because they are caused by the query, not the warehouse. A
  successful query resets the count. While open, the agent says the data warehouse is not
  responding, and that saved reports and the schema overview still work.
- **Schema fallback.** `ria init` and the `bq_schema` tool fall back to a built-in schema if
  BigQuery cannot be read.
- **Long conversations.** Prompts include the last 8 turns (16 messages) verbatim. When the
  stored history passes 18 messages, `begin_turn` folds everything older than the last 16
  into a rolling summary with one model call, so the prompt size stays bounded. If that call
  raises `LLMUnavailable`, `begin_turn` keeps the messages as they are, records the guard
  event `summary:skipped_llm_unavailable`, and lets the turn continue instead of failing it.
- **Best-effort commands.** A `pre_router` node (`graph/prerouter.py`, parsing in
  `graph/commands.py`) runs between `begin_turn` and `router`. It recognises fixed commands
  in code and sets the intent without a model call: `/reports` (list your reports, newest
  first), `/report <id>` (show one report through the MCP tool `reports_get`, author only),
  `/delete <filter>` (`from this conversation` or `conversation`, `all`, or any other text as
  a search, with an optional leading `mentioning` or `about` removed; confirmation as in
  section 3), `/schema` (the cached schema digest, no model), `/good [note]`, `/bad [note]`
  (section 4) and `/help`. No node on these paths calls a model, so they work when both
  models are down. Plain text goes on to the LLM router.
- **Title fallback.** If the titling call in `save_report` raises `LLMUnavailable`, the
  report is still saved, titled with the first 80 characters of the question.
- **Outage scenarios.**

  | Down | Analysis and free-text questions | Commands |
  |---|---|---|
  | Warehouse (breaker open) | Router and SQL generation run; the query is not sent; "The data warehouse is not responding right now..." (outcome `error`). Schema questions, preferences, saving and deleting work. | All work |
  | Both models | The router call fails; "The language model is unavailable right now..." (outcome `error`) | All work, including `/delete` with its confirmation |
  | Warehouse and both models | Same as models down | All work; none of them queries BigQuery (`/schema` uses the schema cached at `ria init`) |

  Saving by free text ("save this as a report") needs the router, so with both models down
  it fails like any other free-text turn. There is no `/save` command.
- **CLI.** `chat_loop` catches every exception from a turn, prints a short message and keeps
  the session. Ctrl-C during a turn cancels that turn only. `ria chat` exits early with a
  clear message if `GEMINI_API_KEY` is unset; `ria doctor` checks the key, Google
  credentials, a BigQuery dry run, and one Gemini call.
- **Cost control.** Flash as the primary model; structured outputs for router and SQL, so no
  calls are wasted on unparseable text; the schema digest is built once at startup from the
  schema cached by `ria init`; Trio embeddings are computed once at `ria init` and only the
  user's question is embedded per turn; at most 50 rows are sent to the analysis prompt.

**Tests.** `tests/test_llm.py`: `test_retries_on_429_then_succeeds`,
`test_falls_back_after_retries_exhausted`, `test_non_retryable_error_goes_straight_to_fallback`,
`test_parse_failures_fall_back_without_marking_primary_down`,
`test_primary_marked_down_is_skipped`, `test_raises_when_both_fail`.
`tests/test_bigquery_runner.py`: `test_breaker_opens_after_threshold_and_cools_down`,
`test_query_classifies_errors_and_trips_breaker`,
`test_classification_of_syntax_auth_and_bytes_billed_errors`, `test_dry_run_reports_bytes_and_cap`.
`tests/test_graph_integration.py`: `test_repair_loop_recovers`,
`test_gives_up_after_repair_budget`, `test_llm_outage_is_graceful`,
`test_context_summary_kicks_in`, `test_commands_work_with_both_models_down`,
`test_delete_command_asks_for_confirmation_with_models_down`,
`test_command_routes_with_overflowing_window_and_models_down`.
`tests/test_graph_foundation.py`: `test_call_llm_enforces_budget`,
`test_pre_router_sets_intent_for_commands_only`, `test_begin_turn_skips_summary_when_model_down`.
`tests/test_commands.py` (parsing of every command). `tests/test_graph_library.py`:
`test_save_report_title_falls_back_to_question_when_model_down`,
`test_list_reports_newest_first_and_empty`, `test_show_report_renders_body_and_sql_or_not_found`.
`tests/test_cli.py`: `test_chat_loop_survives_interrupt_during_turn`.

**Production.** Vertex AI provisioned throughput for the primary model, so 429s become rare
under normal load. A second Vertex region as failover for the model. A queue (Cloud Tasks or
Pub/Sub) in front of the agent when quota is near its limit, with a "your answer is coming"
message instead of an error. The circuit breaker state moves to Memorystore so all instances
share it.

## 6. Quality assurance

**Asks.** Show how the agent's quality is tested and how a reviewer knows it does what the
user meant.

**Prototype.**

- **Unit tests.** 204 tests, no network, under 3 s. Guards, masker, retrieval, preference
  extraction, report target resolution, LLM retry and fallback, BigQuery error
  classification and breaker, trace store and stats, CLI pieces.
- **Graph tests with fakes.** `llm/fake.py` (`FakeLLM`) returns scripted outputs per node
  name; `tools.py` (`FakeToolBox`) returns scripted MCP responses; `golden/embed.py`
  (`FakeEmbedder`) is a deterministic hashing embedder. `tests/test_graph_integration.py`
  runs whole turns through the compiled graph: happy path, repair, give-up, refusal, model
  outage, preference then render, save then delete across a restart, context summary,
  commands with both models down, ratings and corrections.
- **Eval set.** `evals/questions.yaml` has 24 cases across the three demo users: analysis
  questions from the assignment, a scoped user asking about a brand outside their scope
  (Calvin Klein for alice, expects `no_data`), a scoped user asking a users-only question
  ("customers signed up per month", expects `ok` through the semi-join), a PII request, a
  request for individual customers with city and age and a small-group request (both
  expect `ok` within two repairs; the guard rejects or filters the query), a prompt
  injection, an off-topic request, a preference, a save with nothing to save, a delete with
  no matches, and two commands (`/reports`, `/help`). `ria eval`
  (`evals/runner.py`) runs each case on a fresh thread against live Gemini and BigQuery.
- **Deterministic trajectory checks** (`evals/checks.py`, `check_case`), read from the trace:
  intent matches; outcome matches; repairs at most `max_repairs`; each expected table appears
  in the final SQL; no `must_not_match` regex (emails, street names) matches the answer; and
  for analysis turns by a scoped user, the final SQL contains a `brand IN (` predicate
  (direct or inside the semi-join) and every one of the user's brand literals. The column
  comes from `Settings.scope_column`.
- **LLM judge.** The same Gemini client scores each answer 1 to 5 with a one-sentence reason
  (5 = directly answers with a clear takeaway, 1 = wrong, empty or off-topic; an appropriate
  refusal scores 5). A score outside 1 to 5 fails validation and is recorded as a judge failure.
  Judge failures never abort the run.
- **Output.** A results table in the terminal and `evals/results-<timestamp>.json`. Exit code
  1 if any case fails a deterministic check.
- **Tests.** `tests/test_evals.py`: `test_check_case_reports_each_failure`,
  `test_scope_requires_in_predicate`, `test_scope_missing_value_named`,
  `test_scope_accepts_escaped_apostrophe_and_semi_join`, `test_command_cases_pass_with_no_sql`.

**How a reviewer verifies intent.** The deterministic checks prove the trajectory (right
intent, right tables, scope applied, no PII, bounded repairs). The judge covers whether the
answer addresses the question. A reviewer spot-checks a sample each run: read the question,
`ria trace <turn_id>` for the SQL, and the answer, and mark whether the SQL measures what the
executive asked. Disagreements with the judge go into the judge rubric.

**UX evaluation (production).** Thumbs up and down buttons on every answer, stored the way
`/good` and `/bad` already are (on the trace row). Time to first useful answer (from session start to the first answer rated up or saved).
Abandonment rate (sessions that end after a `gave_up`, `no_data` or `error` turn). A monthly
review session with two or three executives, using real transcripts, to find answers that
were correct but not useful.

**CI gate (production).** On every change to prompts, guards, persona or Trios: the unit
suite must pass, every deterministic eval check must pass, and the judge mean must be 4 or
higher. The prototype computes all of these; the judge-mean threshold is not enforced in
code yet.

## 7. Observability

**Asks.** Make each turn inspectable and the system's health measurable.

**Prototype.**

- **Trace schema** (`observability/trace.py`, `TurnTrace`). One row per turn in `traces`:
  `turn_id`, `thread_id`, `user_id`, `intent`, `outcome`, `started_at`, `duration_ms`,
  `tokens_in`, `tokens_out`, `bq_bytes`, `fallback_used`, `repairs`, `rating` (+1, -1 or
  null), `rating_note`, `corrected` (0 or 1), and a JSON payload
  with `node_timings` (node and ms), `llm_calls` (node, model, tokens, latency, fallback
  flag), `sql_attempts` (sql, error, kind, row_count, bytes), `guard_events`, and `error`.
  `rating`, `rating_note` and `corrected` are written later, by `TraceStore.set_rating` and
  `TraceStore.set_corrected`, when the user rates or corrects that answer on a later turn.
  Each model call also gets a row in `llm_ledger`. In `ria chat`, every trace is also
  appended to `traces.jsonl` next to the database.
- **Always written.** `write_trace` (`graph/lifecycle.py`) is the last node on every path,
  including refusals, give-ups, model outages and confirmed or cancelled deletes. The CLI
  prints `[turn <id> · <outcome>]` after each answer.
- **`ria trace <turn_id>`** (`observability/report.py`, `render_trace`) prints a tree: node
  timings, each model call with tokens and a fallback marker, each SQL attempt with its
  error or row count, the guard events, and the rating and correction mark when present.
- **`ria stats [--since]`** (`render_stats`): turns; success rate (outcomes ok, saved,
  deleted, cancelled, listed, shown, rated, help); refusal rate; error rate (error,
  gave_up); fallback rate; mean repairs over analysis turns only; guard rejections by rule
  (`pii`, `demographics`, `scope`, `resolve`, ...); p50 and p95 latency; mean tokens per
  turn; mean BigQuery bytes per turn; `rated good/bad` (counts of +1 and -1 ratings);
  `correction_rate` (corrected analysis turns over analysis turns); turns by intent.
- **LangSmith (opt-in).** Setting `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` in `.env`
  enables LangGraph's built-in LangSmith tracing. There is no project code for it.
- **Tests.** `tests/test_trace.py`: `test_save_get_and_jsonl`, `test_ledger_rows_written`,
  `test_stats`, `test_stats_ratings_and_correction_rate`,
  `test_renderers_show_rating_and_corrected`, `test_render_trace_keeps_bracketed_text_literal`.
  `tests/test_graph_integration.py`: `test_analysis_happy_path` asserts on the trace contents.

**Production.**

- Traces as structured Cloud Logging entries with `turn_id`, `thread_id` and user id as
  labels; Cloud Trace spans per node; LangSmith or Langfuse for model-call detail; a log sink
  to BigQuery for the stats queries.
- Alerts: success rate below 90 % over 1 hour; fallback rate above 20 %; mean repairs per
  analysis turn above 0.5; p95 latency above 20 s; a spike in guard rejections by rule
  (`pii` or `scope` above three times the 7-day baseline), which can mean a prompt regression
  or a probing user.
- Deep dive path: an alert or a user complaint gives a `turn_id`; the trace shows the nodes,
  SQL attempts and guard events; the LangSmith run for the same turn shows full prompts and
  outputs; the BigQuery job id (to be added to the attempt record) opens the job in the
  console with bytes and slot time.

## 8. Persona management

**Asks.** Let non-developers change the agent's tone and instructions without a redeploy.

**Prototype.** `persona/loader.py` (`load_persona`) reads `data/persona.md` on every call to
`Deps.persona()`, which happens each time a router, SQL, repair, analysis or schema prompt is
built. Editing the file changes the next answer with no restart. A missing or empty file
falls back to a one-line default persona. The shipped persona sets tone (lead with the
answer, round numbers, flag thin data) and a weekly emphasis (connect findings to margin, end
with at most three action items with owners).

**Tests.** `tests/test_memory_persona.py`: `test_load_persona`.

**Production.** The persona is a versioned object in GCS (object versioning on). An admin
page shows the current text, lets an authorised editor change it, and previews the change on
a staging thread with a few fixed questions before publishing. Publishing writes a new
version; the agent reads the latest version with a short cache (for example 60 s). Rollback
means selecting an earlier version. The persona version is recorded in each trace, so a
change in answer quality can be tied to a persona edit.

---

## 9. Why these choices

| Choice | Reason |
|---|---|
| LangGraph | The checkpointer and `interrupt()` give the delete confirmation and its restart survival in one node (`confirm_delete`, about 15 lines) and one resume call in the CLI. Typed state and conditional edges keep the safety steps in plain code nodes, not in prompts. Each node is testable with fakes. |
| Gemini 3.8 Flash, fallback 3.5 Flash-Lite | The assignment prefers Gemini. Flash is cheap enough for 3 to 5 calls per turn and supports structured output for routing and SQL. The fallback uses the same key and SDK. |
| MCP for tools | The tool boundary is where permissions live: the graph never imports the BigQuery client. Adding a capability is adding a tool. A reviewer can read the seven tool signatures to see everything the agent can do to the outside world. |
| sqlite-vec | A real KNN index with no server, in the same file as everything else. pgvector would need a Postgres install for the prototype. Vertex AI Vector Search is the production choice; it needs a deployed index endpoint, which is too much setup for a reviewer's laptop. |
| SQLite | One file, no install, and LangGraph ships a SQLite checkpointer. The schema maps directly to Postgres for production. |

**Author's experience.** This is my first LangGraph project. I have built the same kind of
orchestration before without a framework: a master agent handing off to sub-agents, per-call
state, and tool routing, written directly on a voice-agent vendor's SDK and run in
production. In this project LangGraph replaced the parts I had written by hand: the state
container, the routing table, and pause and resume. The graph keeps the structure I used
there: pure-Python guard nodes (SQL guard, scope rewrite, PII masker, delete target
resolution) around three model calls per analysis turn (router, SQL generation, analysis).

## 10. Data flow

The production architecture, one analysis turn, and the delete flow are drawn in
[hld.md](hld.md) sections 2 to 4. The graph wiring is in `graph/build.py`:

```
begin_turn -> pre_router -> one of:
  slash command (no model call):
    /reports:        list_reports
    /report <id>:    show_report
    /delete <filter>: resolve_targets -> confirm_delete (interrupt)
    /schema:         schema_overview
    /good, /bad:     rating
    /help:           help
  plain text:        router -> one of:
    analysis:          retrieve_trios -> generate_sql -> guard_sql -> execute_sql -> analyze -> render
                       (guard_sql and execute_sql route to repair_sql -> guard_sql, or to give_up)
    schema_question:   schema_answer
    report_save:       save_report
    report_delete:     resolve_targets -> confirm_delete (interrupt)
    preference_update: ack_preference
    off_topic:         refuse
-> finalize (mask PII) -> update_memory -> write_trace
Any node that raises LLMUnavailable or BudgetExceeded routes to fail -> finalize.
```

**Model calls per turn**, as built:

| Turn type | Generation calls | Embedding calls |
|---|---|---|
| analysis | 3 (router, generate_sql, analyze) + up to 2 repair_sql | 1 (the question) |
| schema question | 2 (router, schema_answer) | 0 |
| report save | 2 (router, title); 1 if there is no analysis to save | 0 |
| report delete | 1 (router); the resume after confirmation makes no model call | 0 |
| preference update | 1 (router) | 0 |
| off-topic | 1 (router); 0 for an empty message | 0 |
| slash command (`/reports`, `/report`, `/delete`, `/schema`, `/good`, `/bad`, `/help`) | 0 | 0 |
| any turn, once stored history passes 18 messages | +1 (summarize; skipped if both models are down) | 0 |

## 11. Error handling and fallback matrix

| Failure | Detection | Behaviour | User sees | Metric |
|---|---|---|---|---|
| Gemini 429 or 5xx | `is_retryable` in `GeminiClient` | 3 attempts on primary (2 s, 4 s waits), then fallback model; primary skipped for 60 s | Normal answer, slower | `fallback_rate`; `[fallback]` in `ria trace` |
| Gemini 400 or other non-retryable error | Not retryable | Straight to fallback model; primary not marked down | Normal answer | `fallback_rate` |
| Structured output unparseable | `parsed is None` | Up to 3 attempts on the same model, then fallback | Normal answer | `fallback_rate` |
| Both models fail | `LLMUnavailable` | `fail` node, trace written with the error | "The language model is unavailable right now (rate limit or outage). Nothing was lost; please retry shortly." | `error_rate` |
| Both models fail, user types a slash command | `pre_router` matches the command | Command node runs; no model call on the path | Normal command output (report list, report, delete confirmation, schema, rating, help) | outcome `listed`, `shown`, `rated`, `help`, `ok`, `deleted` or `cancelled` |
| Both models fail while history needs summarising | `LLMUnavailable` in `begin_turn` | Summary skipped, messages kept, turn continues | Command output, or the model-unavailable message for free text | guard event `summary:skipped_llm_unavailable` |
| Titling call fails on save | `LLMUnavailable` in `save_report` | Report saved, titled with the question | "Saved report #N: <question>..." | outcome `saved` |
| Per-turn model budget reached | `BudgetExceeded` before the 9th call | `fail` node | "I hit the per-question budget before finishing..." | `error_rate` |
| Off-topic or injection attempt | Router intent `off_topic` | Fixed refusal, no further calls | Refusal with an example of a valid question | `refusal_rate` |
| SQL not a single SELECT, other dataset, unknown column | SQL guard | Repair with the reason, up to 2 times | Answer, or a give-up message listing attempts | `guard_rejections` by rule, `mean_repairs` |
| SQL touches PII | SQL guard `pii` | Repair with the reason | Answer without personal data, or give-up | `guard_rejections.pii` |
| Scoped user's SELECT reads only orders or users | SQL guard scope step | Semi-join through order_items and products added; no repair | Answer filtered to customers or orders of their brands | guard event `scope:semijoin` |
| Customer id and a demographic column in one query | SQL guard `demographics` (Rule A) | Repair with the reason | Aggregated answer without individual rows, or give-up | `guard_rejections.demographics` |
| Group by a demographic column | SQL guard (Rule B) | `HAVING COUNT(DISTINCT <customer id>) >= 10` added; no repair | Groups with fewer than 10 customers are left out; an all-small result goes to the empty-result path | guard event `demographics:min_group` |
| Query above 1 GB | Dry run `bytes_cap` | End turn, no repair | "This question would scan more data than allowed... Narrow the time range or the scope of the question" | outcome `gave_up` |
| BigQuery syntax error | `classify_error` = `syntax` | Repair; breaker not touched | Answer after repair, or give-up | `mean_repairs` |
| BigQuery timeout, quota, unknown | `classify_error` | Repair; counts toward the breaker | Answer after repair, or give-up | `mean_repairs`, `error_rate` |
| Warehouse down | Breaker open after 3 failures, 5 min cooldown | End turn with `circuit_open`, no query sent | "The data warehouse is not responding right now... Saved reports and the schema overview still work." (`/reports`, `/report <id>`, `/schema`) | outcome `error` |
| Empty result | Zero rows | One repair with a loosen-filters hint; second empty ends the turn | "The query ran but returned no rows. Your access covers ..." plus the SQL | outcome `no_data` |
| Repair budget spent | 2 repairs used | `give_up` | "I couldn't get a working query for this after 3 attempts. What I tried: ..." | outcome `gave_up` |
| PII-shaped text in the answer | `mask_pii` regex | Replaced with `[email]`, `[phone]`, `[address]` | Masked answer | not yet recorded |
| Embedding API failure | Exception in `retrieve_trios` | Turn continues with no examples | Normal answer | guard event `retrieval:failed:<type>` |
| Delete with no matching reports | Empty search | No interrupt | "No reports ... were found. Nothing to delete." | outcome `ok` |
| Delete not confirmed | Answer is not `yes`/`y` | Nothing deleted | "Cancelled. Nothing was deleted." | outcome `cancelled` |
| Unexpected exception in a turn (MCP process) | Caught by `chat_loop` | Session continues | "Something went wrong: <type>: <message>" | not traced (gap) |
| Missing key or credentials | `ria chat` / `ria doctor` checks | Exit with instructions | "GEMINI_API_KEY is not set (see .env.example)." | n/a |

## 12. Setup and example run

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

The README has the demo script and the other commands.

### Example run

Transcript to be captured after `ria init` on a machine with credentials; see the demo script in the README.

## 13. Client answers and open items

The first version was built on assumptions (scope by product category, a scoped user's
customer-only questions refused, no rule on demographics). OpsFleet then answered the
clarification questions. The numbers are OpsFleet's question numbers.

| # | Answer | What it changed |
|---|---|---|
| 1a | Access is by brand. The CEO sees everything. The mapping of managers to brands is ours to define. | Scope key is `products.brand` (`Settings.scope_column`). Demo users own brands in `data/users.yaml` (key `brands`); the CEO has `*`. `ria init` warns about brands not in the dataset. Seed Trio 11 is tagged `all`. |
| 1d | Questions outside a manager's brands are out of scope. Customer-level questions (for example signups per month) are answerable, filtered to the manager's data. The CEO is unrestricted. | Orders-only and users-only SELECTs are no longer rejected for scoped users; the guard adds a semi-join through order_items and products (section 2, step 6). |
| 2 | No company-wide churn definition. Default: no purchase in 90 days. State the definition used and let users override it. | `data/definitions.md`, read every turn into the SQL, repair and analysis prompts; answers state the definition; an inline override is used for that answer. Seed Trio 05 uses 90 days. |
| 3 | Revenue is Complete, Shipped and Processing; Cancelled and Returned are excluded; it should be configurable. | Same definitions file. |
| 4 | Hard delete, no recovery window; the confirmation is the safeguard. | No change. |
| 5 | Reports are not shared; only the author views, changes or deletes them. | No change. `/report <id>` also returns only the author's own reports. There is no edit feature. |
| 6 | Demographics only in aggregates, never per individual; define a minimum group size. | Guard Rules A and B with a minimum of 10 customers per group, for every user including the CEO (section 2, step 5). Seed Trio 01 no longer returns demographics. |
| 8a to 8e | All five learning signals are wanted. | `/good` and `/bad` ratings and router-detected corrections, stored on the previous answer's trace and shown in `ria trace` and `ria stats` (section 4). |
| 9 | Timings, alerting and degraded mode are ours to design. | Slash commands handled in code before the router, so reports, schema, ratings and help work with both models down (section 5). |

**Open items, pending with OpsFleet's second reviewer.** Nothing is built for these yet.

- 1b: distribution centers.
- 1c: whether a manager's responsibilities (and so their brands) change.
- 7: multi-language support.
- 9a to 9c: specific resilience requirements.

**Still assumed.** A human analyst reviews Golden Trio candidates before promotion, and no
candidate influences SQL generation without that approval. Concurrency is low for the
prototype (one user per process); production sizing is in the HLD.
