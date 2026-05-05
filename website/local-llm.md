---
title: Running on a local LLM
layout: default
nav_order: 4
permalink: /local-llm/
---

# Running on a local LLM
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

`why` ships with a generic OpenAI-compatible provider that works with Ollama, llama.cpp's `server`, LM Studio, vLLM, and HuggingFace TGI. Useful when you want to keep your code off cloud providers, or just want to try `why` without an API key.

## Quickstart with Ollama (8GB RAM)

```sh
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a small model
ollama pull qwen2.5:3b
# 3. Start the server (Ollama runs on :11434 by default)
ollama serve
```

```sh
export WHY_LLM_PROVIDER=openai-compatible
export WHY_LLM_BASE_URL=http://localhost:11434/v1
why --model qwen2.5:3b src/why/llm.py
```

That's it — `WHY_LLM_MAX_CTX` defaults to `4096` for `openai-compatible`, and `WHY_LLM_NUM_CTX` auto-couples to it, so Ollama's `num_ctx=2048` ceiling is raised to `4096` automatically.

## Recommended local models

| Model         | Params | Download | Fits 8GB RAM | Notes                                  |
|---------------|--------|----------|--------------|----------------------------------------|
| `qwen2.5:3b`  | 3.1B   | ~1.9 GB  | yes          | Strongest instruction-following at 3B. |
| `phi3:mini`   | 3.8B   | ~2.3 GB  | yes          | Good structured-output behavior.       |
| `llama3.2:3b` | 3.2B   | ~2.0 GB  | yes          | Solid generalist.                      |
| `qwen2.5:7b`  | 7.6B   | ~4.7 GB  | tight        | Better quality if you have headroom.   |

## Environment variables

| Variable           | Default                                   | Purpose                                                       |
|--------------------|-------------------------------------------|---------------------------------------------------------------|
| `WHY_LLM_PROVIDER` | `groq`                                    | Set to `openai-compatible` to use a local model.              |
| `WHY_LLM_BASE_URL` | (required for `openai-compatible`)        | OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1`. |
| `WHY_LLM_API_KEY`  | `not-needed`                              | Most local servers ignore this; some require any non-empty.   |
| `WHY_LLM_MAX_CTX`  | `4096` (local), disabled (groq)           | Prompt token budget; oldest commits dropped to fit.           |
| `WHY_LLM_NUM_CTX`  | unset (auto-couples to `WHY_LLM_MAX_CTX`) | Ollama-only: raises model context window above its 2048 default. |

See [Usage](/usage/) for the full semantics of each variable.

## Quality caveat

Local 3B models produce useful drafts but not Groq-70B-quality narratives. Two guardrails kick in automatically when provider is `openai-compatible`:

- **Auto-shrink** drops the oldest commits and truncates long diffs so the prompt fits the context window. A single warning describes what was dropped.
- **Strict citations** are auto-enabled — if the model invents a SHA or PR number, `why` fails loudly with a friendly error rather than emitting hallucinated references. Use `--no-strict-citations` to allow it.

For full semantics of `WHY_LLM_MAX_CTX` (auto-shrink) and `WHY_LLM_NUM_CTX` (Ollama coupling), see [Usage → Auto-shrink](/usage/#auto-shrink-for-small-context-windows) and [Usage → Ollama context window](/usage/#ollama-context-window-num_ctx).
