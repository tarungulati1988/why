---
title: Home
layout: home
nav_order: 1
description: "why — a CLI that explains why code is the way it is."
permalink: /
---

# why
{: .fs-9 }

git blame tells you who. `why` tells you why.
{: .fs-6 .fw-300 }

[Get started](/install/){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/tarungulati1988/why){: .btn .fs-5 .mb-4 .mb-md-0 }

---

`why` is a CLI that explains why code is the way it is by mining git history and PR metadata, then synthesizing it with an LLM.

## Quickstart

```sh
export GROQ_API_KEY=...
why src/path/to/file.py
```

- [Install](/install/) — Homebrew, pip, source, and API key setup
- [Usage](/usage/) — full reference for flags and options
- [Local LLM](/local-llm/) — run offline with Ollama or any OpenAI-compatible server

## Status

WIP. The site grows with each release. Source: <https://github.com/tarungulati1988/why>
