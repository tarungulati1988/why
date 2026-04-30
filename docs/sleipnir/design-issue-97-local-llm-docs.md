# Design — Issue #97: Local LLM setup guide

**Issue:** [#97](https://github.com/tarungulati1988/why/issues/97)
**Branch:** `sleipnir/issue-97-local-llm-docs`
**Depends on:** #93–#96 (all merged) — docs reflect what actually shipped
**Scope:** docs-only

## Decisions

### 1. Placement

Insert as `## Running on a local LLM` between line 46 (end of "After installing — set your API key") and line 48 (start of `## Usage`). Sits as a parallel quickstart to the Groq path. The deeper env-var docs in Usage stay; this section is a focused entry point.

### 2. Drop-in adapted from issue snippet

The issue ships a complete snippet. Adaptations needed to match what shipped:

- `WHY_LLM_NUM_CTX` default row: change "unset" → "unset (auto-couples to `WHY_LLM_MAX_CTX`)" — reflects #95 auto-couple.
- Quality caveat: keep the strict-citations note matching #96's auto-on default (`--no-strict-citations` to opt out).
- Worked example: `why src/why/llm.py --model qwen2.5:3b` is a real file — keeps the example reproducible.

### 3. No tests

Docs-only. The skip-test exception applies. No code paths change.

## File layout

```
README.md   Insert new section between Install and Usage
```

## Strides

```
Stride 1: Insert "Running on a local LLM" section | test: (skip — docs only) | impl: README.md | depends: []
```
