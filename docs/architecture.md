# why — Architecture

`why` is a Python 3.11+ CLI that explains why code is the way it is by mining git history and
PR metadata, then synthesizing an explanation via an LLM. The user points it at a file, line,
or named symbol; it traces the relevant commits, ranks them by explanatory value, and produces
a narrative grounded in diffs and PR descriptions.

## Pipeline at a glance

```
 TARGET SPEC
     |
     v
 [target.py]  parse_target()
     |  Target
     v
 [history.py]  get_file_history() / get_line_history()
     |  list[Commit]   (via git.py → run_git)
     v
 [commit.py]  parse_porcelain()
     |  list[Commit]
     v
 [symbols.py]  find_symbol_range()  ← (symbol targets only; tree-sitter)
     |  (line_range tuple)
     v
 [scoring.py]  select_key_commits()
     |  list[Commit]  (key subset, skipped when --deep)
     v
 [diff.py]  get_commit_diff()   +   [github.py / cache.py]  PR metadata
     |  list[CommitWithPR]                  (sidecar lookups)
     v
 [prompts.py]  build_why_prompt() / build_system_prompt()
     |  list[Message]
     v
 [llm.py]  LLMClient.complete()
     |  raw LLM response string
     v
 [citations.py]  validate_citations()     [timeline.py]  validate_and_repair_timeline()
     |  (strict: raises CitationError;                   (repairs hallucinated SHAs)
     |   non-strict: logs warnings)
     v
 [synth.py] (two_pass=True) → build_grounding_prompt() → LLMClient.complete()
     |  final response string
     v
 [render.py]  render_output()
     |
  STDOUT
```

All orchestration lives in `synth.synthesize_why()`; `cli.main()` wires user flags to it.

## Module map

| Module | Stage | Role | Reads | Writes/Returns |
|---|---|---|---|---|
| `cli.py` | Entry point | Parses CLI flags via Click, wires `LLMClient`, `GitHubClient`, `PRCache`, and delegates to `synthesize_why`; handles error exits | CLI args, env vars | calls `render_output` with final string |
| `target.py` | Input resolution | Parses `TARGET` spec (`path[:line]`) and optional symbol into a `Target` dataclass; enforces path-confinement within repo root | raw spec string, repo `Path` | `Target` |
| `git.py` | I/O | Thin subprocess wrapper around `git`; applies hardening env (`GIT_TERMINAL_PROMPT=0`, `LC_ALL=C`, etc.) and maps exit codes to typed exceptions | `list[str]` args, `cwd` | raw stdout `str` |
| `history.py` | History retrieval | Builds `git log` invocations for file-level (`--follow`) and line-level (`-L`) history; delegates parsing to `commit.parse_porcelain` | `Target.file`, repo `Path`, optional `since` | `list[Commit]` |
| `commit.py` | Parsing | Defines the frozen `Commit` dataclass and `parse_porcelain()`, a single-pass parser for `git log --format=PORCELAIN_FORMAT --shortstat` output | raw `git log` stdout | `list[Commit]` |
| `diff.py` | Diff extraction | Fetches per-commit diffs via `git show` or `git log -L`; truncates to `max_chars` | commit SHA, file `Path`, optional `line_range`, repo `Path` | diff `str` |
| `symbols.py` | Symbol resolution | Uses tree-sitter (Python + Go grammars) to locate a named symbol's line range in a file | `file` `Path`, `symbol` `str` | `(start, end)` line tuple |
| `scoring.py` | Ranking | Pure scoring function `score_commit()` and `select_key_commits()` that pick the most explanatory subset; signals: diff size, message length, keywords, PR presence, recency, merge/junk penalties | `list[Commit]`, PR membership mapping, `date` | ranked `list[Commit]` |
| `timeline.py` | Post-processing | Validates or deterministically regenerates the `## 📊 Timeline` block in LLM output; repairs hallucinated SHAs | LLM response `str`, `list[CommitWithPR]` | repaired response `str` |
| `prompts.py` | Prompt construction | Defines `CommitWithPR`, `PRMetadata`; builds `build_why_prompt()` (user message), `build_system_prompt()` (repo-URL-parameterised), and `build_grounding_prompt()` (two-pass) | `Target`, current code `str`, `list[CommitWithPR]`, repo URL | `list[Message]` |
| `llm.py` | LLM dispatch | `LLMClient` resolves provider (groq / openai-compatible), constructs the appropriate backend, and calls `complete()` with exponential retry; `GroqBackend` lives here | `list[Message]`, system prompt `str` | response `str` |
| `synth.py` | Orchestration | `synthesize_why()` assembles every stage: history → scoring → diffs → PR lookup → prompt → LLM → citations → timeline → optional grounding pass | `Target`, repo `Path`, `LLMClient`, optional `GitHubClient`/`PRCache` | final explanation `str` |
| `render.py` | Output | `render_output()` strips ANSI escapes; prints via `rich.Markdown` on a TTY, plain `click.echo` otherwise | explanation `str`, `color` flag | writes to stdout |
| `cache.py` | Sidecar / I/O | `PRCache` stores `PRMetadata` lists keyed by commit SHA in a JSON file under `~/.cache/why/`; 30-day TTL; atomic writes via temp-file rename | commit SHA `str` | `list[PRMetadata] | None` |
| `citations.py` | Post-processing | `validate_citations()` regex-scans LLM output for SHA and PR-number references not present in the input context; raises `CitationError` when `strict=True` | LLM response `str`, `known_shas`, `known_prs` | `list[ValidationIssue]` (or raises) |
| `github.py` | Sidecar / I/O | `GitHubClient` calls GitHub REST `/commits/{sha}/pulls` to fetch `PRMetadata`; `detect_github_token()` checks `GITHUB_TOKEN` then `gh auth token` | commit SHA `str` | `list[PRMetadata]` |

## Provider backends (src/why/_backends/)

`_backends/` contains the provider-dispatch layer that `LLMClient` delegates to. `base.py`
defines the `Backend` protocol (a single `chat(model, payload, **extra) -> ChatResult` method)
and the `ChatResult` dataclass. `openai_compatible.py` implements `OpenAICompatibleBackend`,
which wraps the `openai` SDK and sends requests to any `/v1/chat/completions` server. When
`num_ctx` is set it injects `extra_body={"options": {"num_ctx": N}}` for Ollama's KV-cache
sizing; other servers ignore the extra body. `GroqBackend` lives in `llm.py` rather than in
this subpackage because Groq is the default provider and was implemented before the backend
abstraction was introduced.

| Module | Backend | Notes |
|---|---|---|
| `_backends/base.py` | Protocol + `ChatResult` | Defines the interface all backends must satisfy |
| `_backends/openai_compatible.py` | `OpenAICompatibleBackend` | Ollama, llama.cpp, LM Studio, vLLM, TGI; optional `num_ctx` for Ollama |

## Key invariants

- Pipeline is unidirectional: cli → target → history → commit → scoring → diff → prompts → llm → citations → timeline → render. No module reaches "back" upstream.
- **No I/O in scoring, timeline, or prompts.** `score_commit`, `select_key_commits`,
  `validate_and_repair_timeline`, `build_why_prompt`, and `build_system_prompt` are pure
  functions over data structures; they call no subprocess, no network, no filesystem.
- **`LLMClient` is the only caller of a remote model.** Backends in `_backends/` are
  dispatch helpers — they translate SDK exceptions to typed `LLMError` subclasses and return
  `ChatResult`. They do not own retry logic; retry lives in `LLMClient.complete()`.
- **Strict citations auto-enable for openai-compatible provider.** `cli.main()` sets
  `strict = (llm.provider == "openai-compatible") and not no_strict_citations`, which causes
  `validate_citations` to raise `CitationError` on any hallucinated SHA or PR reference when
  using a local model. For groq, `strict=False`, so issues are logged rather than raised.
- **`WHY_LLM_MAX_CTX` defaults to 4096 for openai-compatible; disabled for groq.** The
  resolver in `llm._resolve_max_ctx` returns `_DEFAULT_CTX_OPENAI_COMPAT` (4096) when the
  env var is unset and provider is `"openai-compatible"`, and `None` (disabled) for groq.
  Setting `WHY_LLM_MAX_CTX=0` explicitly disables auto-shrink for any provider.
- **`WHY_LLM_NUM_CTX` auto-couples to `WHY_LLM_MAX_CTX`.** When `WHY_LLM_NUM_CTX` is
  unset, `LLMClient` passes the resolved `_resolve_max_ctx` value as `num_ctx` to
  `OpenAICompatibleBackend`, coupling Ollama's KV-cache window to the prompt budget.
- **`synthesize_why` is the single integration seam.** All pipeline stages are invoked from
  this function in `synth.py`; `cli.main()` calls nothing else from the pipeline directly.

## Where features live

| Feature | Module(s) |
|---|---|
| `--brief` | `cli.py` (flag), `prompts.build_why_prompt` (`_BRIEF_TAIL` appended to user message) |
| `--verify` (two-pass grounding) | `cli.py` (flag → `two_pass=True`), `synth.synthesize_why` (second `llm.complete` call), `prompts.build_grounding_prompt`, `prompts.GROUNDING_SYSTEM_PROMPT` |
| `--deep` / `--max-commits` | `cli.py` (flags), `synth.synthesize_why` (bypasses `select_key_commits`, applies `max_commits` cap) |
| Strict citations | `cli.py` (auto-set `strict` flag), `citations.validate_citations` (raises `CitationError`), `synth.synthesize_why` (passes `strict`) |
| Auto-shrink (`WHY_LLM_MAX_CTX`) | `llm._resolve_max_ctx`, `synth._shrink_for_budget`, `synth.synthesize_why` |
| Ollama `num_ctx` coupling | `llm.LLMClient.__init__` (resolves `num_ctx`), `_backends/openai_compatible.OpenAICompatibleBackend` (injects `extra_body`) |
| Symbol scoping (tree-sitter) | `symbols.find_symbol_range` (`.py` and `.go` grammars), `synth._resolve_line_range` |
| GitHub PR metadata + cache | `github.GitHubClient`, `cache.PRCache`, `synth.synthesize_why` (cache-first lookup loop) |
| Timeline validation / repair | `timeline.validate_and_repair_timeline`, `timeline.render_deterministic_timeline` |

## Data shapes

- **`Target`** (`target.py`) — frozen dataclass: `file: Path`, `line: int | None`,
  `symbol: str | None`. The canonical input to every pipeline stage after CLI parsing.
- **`Commit`** (`commit.py`) — frozen dataclass: SHA, author, date, subject, body, parents
  tuple, additions/deletions int. The unit returned by `parse_porcelain` and consumed by
  scoring, diff, and prompt stages.
- **`CommitWithPR`** (`prompts.py`) — frozen dataclass pairing a `Commit` with optional
  `pr_body`, `pr_number`, `pr_title`, and `diff` text. The enriched unit consumed by prompt
  construction and post-processing.
- **`PRMetadata`** (`prompts.py`) — `NamedTuple`: `number: int`, `title: str`, `body: str`.
  Returned by `GitHubClient` and `PRCache`; stored as the value in `resolved_prs` dict.
- **`Message`** (`llm.py`) — dataclass: `role: Literal["user", "assistant"]`, `content: str`.
  The wire type passed to `LLMClient.complete()` and serialized to the provider's payload.
- **`ChatResult`** (`_backends/base.py`) — dataclass: `content: str`, optional token counts.
  Returned by every backend `chat()` call; token fields are `None` when the server omits usage.

## Design docs

Decision history lives in `docs/designs/` (per-feature design documents from project
inception) and `docs/sleipnir/` (Sleipnir agent design issues, numbered by GitHub issue).
The master idea document that motivated the project is
`docs/design/why-idea-04-18-2026-1.0.0.md`.
