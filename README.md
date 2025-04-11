# suss

**Find bugs in your code changes with a codebase-aware agent.**

Just run `suss` in your terminal to get a bug report in under a minute.

![Demo](demo.gif)

`suss` analyzes the diff between your local and remote branch. For each code change, an AI agent gathers context on the change from the rest of the codebase. That context is used to understand the impact of the code change and scan it for bugs.

## Installation

```bash
> pipx install suss
```

Once installed, you must choose an LLM provider. To use OpenAI or Anthropic, just add your API key to your environment:

```bash
> export OPENAI_API_KEY="..." # For GPT
> export ANTHROPIC_API_KEY="..." # For Claude
```

You can also use other models, including local ones. `suss` wraps LiteLLM, so you can use any model listed [here.](https://docs.litellm.ai/docs/providers)

## Usage

Run `suss` in the root directory of your codebase.

By default, it analyzes every file that's different locally from your remote branch. You can see these files by running `git status`.

To run `suss` on a specific file, you can do:

```bash
> suss --file="path/to/file.py"
```

Alternatively, you can set up a pre-commit hook to automatically run `suss` before you push a commit. Think of it as an automated code review before your code reaches GitHub.
