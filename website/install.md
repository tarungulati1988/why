---
title: Install
layout: default
nav_order: 2
permalink: /install/
---

# Install
{: .no_toc }

## Table of contents
{: .no_toc .text-delta }

1. TOC
{:toc}

---

## Homebrew (macOS — recommended)

```sh
brew tap tarungulati1988/why
brew install why
```

> **Note:** The `tarungulati1988/why` tap must be set up first — see [homebrew-why](https://github.com/tarungulati1988/homebrew-why). This becomes available after the first tagged release.

## pip / uv

```sh
pip install git-why
# or
uv tool install git-why
```

The PyPI package is `git-why`; the CLI command installed is `why`.

## From source

```sh
git clone https://github.com/tarungulati1988/why.git
cd why
pip install -e ".[dev]"
```

See the [README](https://github.com/tarungulati1988/why#development) for the full local setup.

## After installing — set your API key

```sh
export GROQ_API_KEY=your_key_here
```

`why` uses Groq as the default LLM provider. Get a free API key at [console.groq.com](https://console.groq.com).

For offline / local LLM use, see [Running on a local LLM](/local-llm/) — no API key required.
