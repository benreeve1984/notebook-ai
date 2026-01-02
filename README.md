# notebook-ai

Bring Claude into your Jupyter notebook for collaborative data analysis.

## Why?

LLMs have transformed how we write code, but getting real value from them requires close collaboration—not just copy-pasting from a chatbot or hoping an autonomous agent figures things out.

**The problem with chatbots:** They can't see your data or run your code. You end up describing your DataFrame in words, copying error messages back and forth, and never quite getting code that works first time.

**The problem with agents:** They work autonomously but often go off-piste. You hand over control and hope for the best, then spend time unpicking what they did (or didn't do).

**The notebook-ai approach:** Work cell-by-cell in dialogue with the LLM. You stay in control, building up your analysis step by step. The LLM sees your full notebook context—executed code, outputs, variables you choose to share—so it can give genuinely useful help. And with tools, it can actually *do* things on your behalf, not just suggest code.

This unlocks powerful workflows:
- **Get unstuck**: Ask for a code snippet when you can't remember the pandas syntax
- **Review as you go**: Have Claude check your logic before you move on
- **Explore data**: Let Claude summarise a DataFrame or suggest what to look at next
- **Build skills**: Learn by doing, with an expert looking over your shoulder

## Installation

```bash
pip install "git+https://github.com/benreeve1984/notebook-ai.git"
```

## Setup

Create a `.env` file in your project directory:

```bash
ANTHROPIC_API_KEY=sk-ant-...
```

The API key is loaded automatically when you run `%load_ext notebook_ai`.

## Quick Start

```python
# Cell 1: Load the extension
%load_ext notebook_ai

# Cell 2: Your normal analysis code
import pandas as pd

sales = pd.DataFrame({
    'product': ['Widget A', 'Widget B', 'Widget C'],
    'units': [150, 89, 203],
    'price': [19.99, 34.50, 9.99]
})
sales['revenue'] = sales['units'] * sales['price']
```

```python
# Cell 3: Ask Claude about your data
%%prompt
What's the total revenue in {sales}? Which product is performing best and why might that be?
```

That's it. Claude sees your notebook context and the `sales` DataFrame, and gives you a useful answer.

## Core Concepts

### Sharing Variables with `{variable}`

Use curly braces to give Claude access to specific variables:

```python
%%prompt
What patterns do you see in {monthly_data}?
Are there any outliers I should investigate?
```

Claude sees the variable's contents and can reason about it. You control exactly what context to share.

### Giving Claude Tools with `@tool`

Sometimes you want Claude to *do* something, not just talk about it. Register functions as tools:

```python
from notebook_ai import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city using Open Meteo API."""
    # ... implementation ...
    return f"Weather in {city}: {temp}°C, {condition}"
```

Then reference the tool in your prompt:

```python
%%prompt
Check the weather in London and Edinburgh using {get_weather}.
Which city is warmer today?
```

Claude will call your function and use the results to answer. Tools must use simple parameter types (`str`, `int`, `float`, `bool`).

**Tip:** For tools that work with DataFrames, use `str` as the type hint for variable names—they get resolved to the actual objects automatically:

```python
@tool
def top_products(df: str, n: int = 3) -> str:
    """Return top N products by revenue."""
    return df.nlargest(n, 'revenue').to_string()
```

## Practical Examples

### Get Unstuck with Syntax

```python
%%prompt
I have {df} and want to group by 'region', calculate the mean of 'value',
then sort descending. What's the pandas code for this?
```

### Review Your Code

```python
def calculate_churn(customers_df):
    # ... your implementation ...
```

```python
%%prompt --code
Review the calculate_churn function above. Are there any edge cases
I'm missing? Return an improved version.
```

### Explore Your Data

```python
%%prompt
Summarise {experiment_results}. What are the key findings?
What follow-up analysis would you recommend?
```

### Build a Quick Tool

```python
@tool
def run_statistical_test(column_a: str, column_b: str) -> str:
    """Run a t-test between two columns of the current dataset."""
    from scipy import stats
    result = stats.ttest_ind(df[column_a], df[column_b])
    return f"t-statistic: {result.statistic:.3f}, p-value: {result.pvalue:.4f}"
```

```python
%%prompt
Use {run_statistical_test} to compare 'control_group' and 'treatment_group'.
Is the difference statistically significant?
```

## Options

### Code-Only Mode

When you just want code to copy-paste, use `--code`:

```python
%%prompt --code
Improve the `process_data` function above: better names, type hints, more Pythonic.
```

Returns clean code without explanation—ready to paste into the next cell.

### Caching

Responses are cached automatically. Re-running "Run All" returns cached results instantly (marked with *"cached response"*). This means your notebook stays reproducible without repeated API calls.

```python
%%prompt --no-cache
Roll the dice using {roll_dice}
```

Use `--no-cache` to force a fresh API call when you need it.

```python
from notebook_ai import clear_cache, cache_stats

clear_cache()    # Reset all cached responses
cache_stats()    # See cache statistics
```

### Model Selection

```python
%%prompt model=claude-sonnet-4-20250514
Use a faster model for simple questions
```

### Quiet Mode

```python
%%prompt quiet
Don't show which tools were called in the output
```

## How It Works

1. **Context Building**: Collects your executed cells and their outputs
2. **Reference Parsing**: Extracts `{variable}` and `{tool}` references from your prompt
3. **Variable Resolution**: Converts referenced variables to string representations
4. **Tool Preparation**: Makes `@tool` functions available to Claude
5. **LLM Call**: Sends prompt with full context to Claude via Claudette
6. **Tool Loop**: If Claude calls tools, executes them with resolved arguments
7. **Response**: Renders the final response as Markdown in cell output

## Tips for Data Analysts

**Start small.** Use `%%prompt` for quick questions as you work. "What does this error mean?" or "How do I reshape this DataFrame?"

**Share context deliberately.** Only reference variables Claude needs with `{variable}`. This keeps prompts focused and responses relevant.

**Build up your toolkit.** Create `@tool` functions for repeated operations—statistical tests, data validation, API calls. Claude can then orchestrate them for you.

**Use `--code` liberally.** When you want implementation not explanation, `--code` gives you clean, copy-pasteable results.

**Review incrementally.** After writing a function, ask Claude to review it before moving on. Catching issues early saves debugging later.

**Trust but verify.** Claude is helpful but not infallible. Check the code it suggests, especially for edge cases.

## Roadmap

Features we're planning to add:

- **Streaming responses** — See Claude's response as it's generated, rather than waiting for the full reply
- **Response modes** — Different modes for different workflows, e.g. a "learning" mode that explains concepts as it goes, or a "terse" mode for experienced users
- **One-click code copying** — Make it even simpler to grab code snippets from responses and paste them into new cells
- **Pluggable LLM backends** — Swap Claudette for your enterprise LLM wrapper, so you can use notebook-ai at work with approved infrastructure

Have ideas? Open an issue.

## Development

```bash
git clone https://github.com/your-org/notebook-ai.git
cd notebook-ai
pip install -e ".[dev]"
pytest
```

## License

MIT
