# suss

**Check your code for bugs with a codebase-aware agent.**

Just run `suss` in your terminal to get a bug report in under a minute.

![Demo](demo.gif)

`suss` analyzes the diff between your local and remote branch. For each code change, an AI agent gathers context on how it interacts with the rest of the codebase. The agent then audits the code change for bugs, including downstream effects on other code.

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

By default, it analyzes every [code file](https://github.com/shobrook/suss/blob/master/suss/constants/languages.py) that's new or modified compared to your remote branch. These are the same files you see when you run `git status`.

To run `suss` on a specific file:

```bash
> suss --file="path/to/file.py"
```
