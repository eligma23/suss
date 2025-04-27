# suss

**Find bugs with a codebase-aware AI agent.**

Suss looks for bugs in your local code changes, i.e. the diff between your local and remote branch. Just run `suss` in your terminal to get a bug report in under a minute.

![Demo](demo.gif)

For each code change, an AI agent gathers context on how it interacts with the rest of the codebase. Then the agent audits the change and its downstream effects on other code.

Think of suss as a quick and easy sanity check before pushing your code.

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

**Optional:** You can boost performance using the [Cohere reranker model.](https://cohere.com/rerank) Just add your Cohere API key to your environment:

```bash
> export COHERE_API_KEY="..."
```

## Usage

Run `suss` in the root directory of your codebase.

By default, it analyzes every [code file](https://github.com/shobrook/suss/blob/master/suss/constants/languages.py) that's new or modified compared to your remote branch. These are the same files you see when you run `git status`.

To run `suss` on a specific file:

```bash
> suss --file="path/to/file.py"
```
