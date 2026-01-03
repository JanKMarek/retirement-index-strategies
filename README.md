# Retirement Index Strategies

This project implements various retirement index strategies.

i/ Baseline: IXIC buy and hold. 

How to improve? 
- select a subset of stocks with stronger growth: VGT, IPO
- select individual stocks 
- protect against drawdown and only stay in the market when it's doing well
- try to predict market's future using other asset classes or VIX


i/ Investing in the index with several filters/leverage options: 
  - signal: index above 200d/50d MA and (optionally) VIX above threshold
  - underlying uses different levels of leverage: QQQ, QLD, TQQQ

  Produces CAGR, MaxDrawdown, #trades, Equity Curve Chart with Monthly/Annual strategy returns

Results: 
- the VIX filter has minimal impact, so don't use it. 
- using 50d MA reduces CAGR and reduces the max dd, somewhere between using 2x and 3x leverage. There is a spectrum
-    of CAGR/maxDD combinations, select based on the risk appetite:  

CAGR, Max Drawdown, Long Entries, Strategy Parameters
16%, -22%, 34, {'ma_days': 200, 'use_vix_filter': False, 'vix_threshold': 30.0, 'trading_instrument': 'QQQ'}
28%, -40%, 34, {'ma_days': 200, 'use_vix_filter': False, 'vix_threshold': 30.0, 'trading_instrument': 'QLD'}
40%, -55%, 34, {'ma_days': 200, 'use_vix_filter': False, 'vix_threshold': 30.0, 'trading_instrument': 'TQQQ'}






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
