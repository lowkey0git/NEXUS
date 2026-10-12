# Personal Assistant

A general-purpose assistant that sits on a hosted model (Groq's free tier by default; Anthropic and any OpenAI-compatible endpoint also work) and adds what the model doesn't have by itself: accounts, conversation history, file handling, real tools, backend-enforced permissions, approvals, and verification of what actually happened.

Two ways to ask for help, chosen per message in the composer:

| Mode | What it does |
|---|---|
| **Advise** | Research and preparation only. Action tools are not even offered to the model, and the backend refuses them if called. |
| **Auto** | The model decides: answer, research, checklist, or do the work. |
| **Do it** | The model uses its tools to carry the task out. Consequential steps still pause for your approval. |

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY (free key: https://console.groq.com/keys)
python -m app               # http://127.0.0.1:8000
```

Docker: `docker build -t assistant . && docker run -p 8000:8000 -v assistant-data:/data --env-file .env assistant`

Create an account in the browser (data is per user, stored under `ASSISTANT_DATA_DIR`). Settings → shows exactly which tools are available and which are not, and why.

Providers: `ASSISTANT_PROVIDER=groq` (default, `llama-3.3-70b-versatile`), `anthropic` (set `ANTHROPIC_API_KEY`), or `openai_compat` (`OPENAI_API_KEY`, `OPENAI_BASE_URL`). Override the model with `ASSISTANT_MODEL`.

Tests: `python -m pytest tests -q` (66 tests, no network or API key needed).
Browser test: `pip install playwright && playwright install chromium && python tests/ui_smoke.py ./shots`

## Streaming

Replies appear as the model writes them. The run loop publishes text deltas to an in-memory hub, `GET /api/runs/{id}/stream` relays them as server-sent events, and the UI shows them live until the stored message replaces them at the end of each model step. The stream is only a view: messages, steps and approvals are still persisted and polled, so if the stream drops or the server restarts nothing is lost and the UI keeps working.

- **Groq / OpenAI-compatible:** true token streaming. Tool-call fragments are reassembled by index before anything executes, so tools only ever run on complete, schema-validated arguments.
- **Anthropic:** intentionally *not* streamed on the wire (the whole reply arrives at once through the same path). Streaming it means reassembling server-side web-search blocks and `pause_turn` continuations, which I couldn't verify without a live key. Settings shows which mode is active.
- If a connection drops after text was already shown, the run fails with a clear message instead of silently replaying; "Try again" resumes.

## What it can do today

| Tool | Risk | Notes |
|---|---|---|
| `web_search` | read | Provider-hosted search (Anthropic only). Cited by the model. |
| `web_fetch` | read | Reads web pages/PDFs. SSRF-guarded (private/loopback/link-local blocked on every redirect hop). Output is wrapped as untrusted data. |
| `read_document` | read | Uploaded or generated txt, md, csv, json, pdf (text layer), docx, xlsx. Paged. |
| `list_files`, `calculator` | read | Calculator is a safe AST evaluator: totals are computed in code, not by the model. |
| `write_text_file`, `create_spreadsheet` | write | Create downloadable .md/.txt/.csv/.json/.xlsx. Each is re-read from disk and verified before the tool reports success. |
| `delete_file` | consequential | Needs approval. |
| `send_email` | consequential | Needs approval. Only appears when `SMTP_HOST` + `SMTP_FROM` are set. Reports "accepted by the mail server", not "delivered". |

Real automation workflows that work end to end: *read uploaded PDFs/Word/Excel/CSV → extract → compute → produce a spreadsheet or report*, and *fetch several pages → compile into a spreadsheet or checklist*.

## How the guarantees are enforced

- **Permissions live in `app/tools/registry.py::execute`**, the only path to any tool. It checks mode, validates arguments against the tool's JSON schema, and for consequential tools atomically consumes a single-use approval that is bound to the exact run, tool call id and arguments. Changing the arguments, reusing an approval, or deciding someone else's approval all fail (tests: `test_permissions.py`).
- **Honest results.** Tool output (including failures, denials and verification evidence) is fed back to the model, and the system prompt forbids claiming completion without a confirming result. The UI shows the step log from stored events, not from model prose.
- **Untrusted content.** Web pages and files are wrapped in an "untrusted data" envelope, and even a model that is fooled into calling a destructive tool still hits the approval gate (`test_prompt_injection_...`).
- **Durability.** Runs, events, approvals and messages are persisted in SQLite. Cancel works mid-run and while waiting for approval. A crash or restart marks in-flight runs `interrupted` (nothing executes without a live approval), and *Continue* resumes from history with unfinished tool calls repaired as "did not complete; verify before repeating".
- **Provider resilience.** Retries with backoff on 429/5xx/timeouts; side-effecting tools are never auto-retried.

## What has and hasn't been verified

Verified by automated tests (66 + a browser test): auth and tenant isolation, file upload limits and path safety, every tool, SSRF blocking, the approval gate, cancel/continue/restart behavior, the full run loop over HTTP, the Anthropic and OpenAI-compatible adapters' request/response handling and the streaming parser (against simulated HTTP/SSE: split tool-call arguments, parallel calls, dropped connections, rate limits, Groq's `tool_use_failed`), the SSE endpoint end to end, and the browser UI (chat, text visibly growing while streaming without duplication, markdown/XSS safety, attach, approve, download, failure + retry, mobile layout).

**Not verified:** behavior against the live Groq, Anthropic or OpenAI APIs. The development environment had no API key and could not reach api.groq.com, so the model in every test is a scripted stand-in and the streaming parser was tested on simulated streams written from the documented OpenAI-style SSE format. Before relying on it, set your key and run a real conversation (try the three example prompts on the home screen).

**Groq free-tier notes:** it enforces per-minute token and request limits, so long files can trigger a 429 (the app retries with backoff and says what to lower: `ASSISTANT_TOOL_OUTPUT_CHARS`). Open-weight models sometimes emit malformed tool calls; the adapter retries those and otherwise reports it. Expect weaker multi-step tool use than a frontier model, so keep requests focused or choose a larger model in `ASSISTANT_MODEL`. The native `web_search` tool type is `web_search_20250305`; if your account uses a newer version, change it in `app/providers/anthropic.py`.

## Known limitations

- Not built: browser automation (Playwright), scheduled/recurring workflows, Gmail/Calendar integrations, image/screenshot input, Anthropic wire-level streaming, password reset, email verification.
- A run waiting for approval lives in server memory: restart loses the wait (the run is marked interrupted; nothing executes).
- The tool timeout stops *waiting*, not the underlying thread. Write tools finish in milliseconds today, but a future long-running tool should run in a subprocess.
- `web_fetch` resolves DNS twice (check, then connect). For hostile multi-tenant use, route through an egress proxy.
- SQLite and in-process workers suit one server process. Moving to PostgreSQL + a queue is a change inside `app/db.py` and `app/main.py::start_run`.
- Login throttling is in-memory per process.
- Scanned (image-only) PDFs have no extractable text and are reported as such.
- The assistant cannot perform physical-world actions; for errands like a bank visit it prepares you (requirements, documents, what to confirm by phone) and says what remains unverified.

## Layout

```
app/main.py            HTTP API + static UI          app/agent.py       run loop, system prompt, approvals wait
app/tools/registry.py  permission choke-point        app/tools/builtin.py  the tools
app/providers/         Anthropic + OpenAI-compatible  app/net.py        SSRF-safe fetch
app/streaming.py       live-text hub for SSE
app/db.py, filestore.py, auth.py, config.py          app/static/index.html  the UI
tests/                 unit, API end-to-end, UI smoke
```

## Next steps worth doing

1. Run it against a real model and tune `build_system_prompt` from what you see.
2. Playwright tool (read-only first: open page, extract, screenshot), then form-filling behind approvals.
3. Scheduled workflows (store a prompt + schedule; each firing is a normal run in a new conversation).
4. Anthropic streaming (once testable with a key), PostgreSQL, and an out-of-process worker when you outgrow one box.
