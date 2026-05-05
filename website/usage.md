---
title: Usage
layout: default
nav_order: 3
permalink: /usage/
---

# Usage
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Prerequisites

Set `GROQ_API_KEY` before running (see [Install](/install/)).

By default `why` uses Groq as the LLM provider. The active provider is controlled by `WHY_LLM_PROVIDER` (default: `groq`). Two providers are currently supported:

- **`groq`** (default) — Groq cloud API; requires `GROQ_API_KEY`.
- **`openai-compatible`** — any OpenAI-compatible local server (Ollama, llama.cpp, LM Studio, vLLM, TGI, …); requires `WHY_LLM_BASE_URL`. Example with Ollama:

  ```sh
  WHY_LLM_PROVIDER=openai-compatible \
  WHY_LLM_BASE_URL=http://localhost:11434/v1 \
  why --model qwen2.5:3b ...
  ```

### Auto-shrink for small context windows

Local 3B-class models often have small context windows (Ollama's default `num_ctx` is 2048). `why` auto-shrinks the prompt to fit:

- **`WHY_LLM_MAX_CTX`** — token budget for prompt assembly. Set to a positive integer to enable; set to `0` to disable.
- When unset, defaults to **`4096` for `openai-compatible`** providers and is **disabled for `groq`**.
- When shrinking fires, `why` drops the oldest commits first and truncates each remaining diff to 80 lines, then prints a single warning to stderr describing what was dropped.
- `--max-commits` is honored *before* shrinking (user cap wins).

### Ollama context window (`num_ctx`)

`WHY_LLM_MAX_CTX` controls how much context `why` assembles client-side. `WHY_LLM_NUM_CTX` tells Ollama how large a KV cache to allocate server-side. They are kept in sync automatically — usually you only need to set `WHY_LLM_MAX_CTX`.

Ollama defaults to `num_ctx=2048` regardless of the model's actual capability. Other OpenAI-compatible servers (vLLM, llama.cpp, LM Studio, TGI) ignore unknown options and are unaffected.

- **`WHY_LLM_NUM_CTX`** — when set on `openai-compatible`, passed to the server as `extra_body={"options": {"num_ctx": N}}`. Must be a positive integer; `0`, negative, or non-integer values raise an error at startup.
- **Auto-couple to `WHY_LLM_MAX_CTX`** — when `WHY_LLM_NUM_CTX` is unset and a `WHY_LLM_MAX_CTX` is in effect (whether user-set or the auto-default `4096`), `why` uses that value as `num_ctx`. The auto-shrink budget and the server's allocated context stay matched without manual bookkeeping. (So with stock settings on openai-compatible, num_ctx is sent as 4096.)
- **Disable** — set `WHY_LLM_NUM_CTX` explicitly to override the auto-couple, or set `WHY_LLM_MAX_CTX=0` to disable shrinking entirely (which also disables auto-couple).
- **No effect on `groq`** — silently ignored.

## GitHub token (optional but recommended)

`why` fetches PR metadata from the GitHub API. It discovers a token automatically:

1. `GITHUB_TOKEN` env var — set this if you prefer explicit control
2. `gh` CLI — if you have [GitHub CLI](https://cli.github.com) installed and have run `gh auth login`, `why` picks up the token automatically
3. Unauthenticated — works for public repos but is rate-limited to 60 requests/hour

For private repos or heavy use, set a token explicitly:

```sh
export GITHUB_TOKEN=your_personal_access_token
```

A classic PAT with `repo` (read) scope is sufficient.

## Analyse a file

```bash
why src/auth/middleware.py
```

## Narrow to a line

```bash
why src/auth/middleware.py:42
```

## Narrow to a symbol

```bash
why src/auth/middleware.py handle_session_timeout
```

## Override the model

```bash
why --model llama-3.1-8b-instant src/auth/middleware.py
```

## Get a 3-sentence summary (`--brief`)

```bash
why --brief src/auth/middleware.py        # concise mode: one paragraph
```

## Verify claim grounding (`--verify`)

```bash
why --verify src/auth/middleware.py       # second LLM call checks intent claims
```

## Disable color / pipe-friendly output (`--no-color`)

```bash
why --no-color src/auth/middleware.py     # raw markdown, no ANSI
why src/auth/middleware.py | grep "added" # piping auto-disables rich
```

## Get help

```bash
why --help          # full reference with argument descriptions
why --version       # print version and exit
```
