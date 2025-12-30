# Retirement Index Strategies

This project implements various retirement index strategies.

i/ Investing in the index with several filters/leverage options: 
  - signal: index above 200d/50d MA and (optionally) VIX above threshold
  - underlying uses different levels of leverage: QQQ, QLD, TQQQ

  Produces CAGR, MaxDrawdown, Equity Curve Chart, Monthly/Annual strategy returns


## Installation

```bash
uv sync
```

## Usage

```bash
uv run retirement-index-strategies
```

## Testing

```bash
uv run pytest
```
