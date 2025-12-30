import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots


def run_advanced_backtest(backtest_start_date='2000-01-01', ma_days=200, use_vix_filter=True, display_chart=True, strategy_underlying='QLD', vix_threshold=30.0):
    # 1. Data Fetching (25 Years)
    start_date = '1999-01-01'
    print("Fetching data...")
    # ^NDX = Nasdaq 100, ^VIX = Volatility Index
    tickers = ['^NDX', '^VIX', strategy_underlying]
    data = yf.download(tickers, start=start_date, progress=False)

    # Data Cleaning: Handle MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        df = data.xs('Close', level=0, axis=1) if 'Close' in data.columns.levels[0] else data['Close']
    else:
        df = data['Close']

    # Rename columns for clarity
    df = df.rename(columns={'^NDX': 'NDX', '^VIX': 'VIX'})
    df = df.dropna(subset=['NDX'])  # Ensure index data exists

    # 2. Indicator Calculation
    # Use the ma_days parameter for the moving average
    sma_col = f'SMA{ma_days}'
    df[sma_col] = df['NDX'].rolling(window=ma_days).mean()

    # Fill VIX gaps (VIX data can be spotty in very early years, though robust post-2000)
    df['VIX'] = df['VIX'].ffill()

    # 3. Signal Generation
    # Signal is true when NDX is above its SMA
    signal_condition = (df['NDX'] > df[sma_col])
    if use_vix_filter:
        # If the VIX filter is used, the signal is only true if VIX is also below the threshold
        signal_condition &= (df['VIX'] < vix_threshold)
    df['Signal'] = signal_condition.astype(int).shift(1) # Shift to prevent lookahead bias


    # 4. Return Calculation
    df['NDX_Ret'] = df['NDX'].pct_change()
    df[f'{strategy_underlying}_Real_Ret'] = df[strategy_underlying].pct_change()

    # Synthetic Return (e.g., for QLD Pre-2006): 2x NDX Return - approx daily expense ratio (0.95%/252)
    daily_expense = 0.0095 / 252
    df[f'{strategy_underlying}_Synth_Ret'] = (2 * df['NDX_Ret']) - daily_expense

    # Combine Real and Synthetic returns
    df['Asset_Ret'] = df[f'{strategy_underlying}_Real_Ret'].combine_first(df[f'{strategy_underlying}_Synth_Ret'])

    # Strategy Return: Signal (0 or 1) * Asset Return
    df['Strategy_Ret'] = df['Signal'] * df['Asset_Ret']

    # Filter for analysis period
    res = df.loc[backtest_start_date:].copy()

    # 5. Performance Metrics
    initial_capital = 100000
    n_years = (res.index[-1] - res.index[0]).days / 365.25

    # Strategy Metrics
    res['Equity'] = initial_capital * (1 + res['Strategy_Ret']).cumprod()
    cagr_strategy = (res['Equity'].iloc[-1] / initial_capital) ** (1 / n_years) - 1
    res['Peak_Strategy'] = res['Equity'].cummax()
    res['Drawdown_Strategy'] = (res['Equity'] - res['Peak_Strategy']) / res['Peak_Strategy']
    max_dd_strategy = res['Drawdown_Strategy'].min()
    max_dd_date_strategy = res['Drawdown_Strategy'].idxmin()

    # Nasdaq 100 (Buy & Hold) Metrics
    res['NDX_Rebased'] = initial_capital * (1 + res['NDX_Ret']).cumprod()
    cagr_ndx = (res['NDX_Rebased'].iloc[-1] / initial_capital) ** (1 / n_years) - 1
    res['Peak_NDX'] = res['NDX_Rebased'].cummax()
    res['Drawdown_NDX'] = (res['NDX_Rebased'] - res['Peak_NDX']) / res['Peak_NDX']
    max_dd_ndx = res['Drawdown_NDX'].min()
    max_dd_date_ndx = res['Drawdown_NDX'].idxmin()

    # 6. Reporting
    print(f"--- Backtest Report ({backtest_start_date}-Present) ---")
    print(f"\n--- Strategy ({strategy_underlying} + Filters) ---")
    print(f"Final Portfolio Value: ${res['Equity'].iloc[-1]:,.0f}")
    print(f"CAGR: {cagr_strategy:.2%}")
    print(f"Max Drawdown: {max_dd_strategy:.2%} (Occurred: {max_dd_date_strategy.date()})")

    print("\n--- Nasdaq 100 (Buy & Hold) ---")
    print(f"Final Portfolio Value: ${res['NDX_Rebased'].iloc[-1]:,.0f}")
    print(f"CAGR: {cagr_ndx:.2%}")
    print(f"Max Drawdown: {max_dd_ndx:.2%} (Occurred: {max_dd_date_ndx.date()})")


    # 7. Plotting with Plotly
    if display_chart:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.4, 0.6],
            specs=[[{"secondary_y": True}], [{"type": "table"}]],
            subplot_titles=("", "Strategy Monthly Returns")
        )

        # Trace 1: Portfolio Equity (Log Scale recommended for growth curves)
        fig.add_trace(
            go.Scatter(x=res.index, y=res['Equity'], name=f"Strategy ({strategy_underlying} + Filters)",
                       line=dict(color='green', width=2)),
            secondary_y=False, row=1, col=1
        )

        # Trace 2: Nasdaq 100 Index (Benchmark)
        fig.add_trace(
            go.Scatter(x=res.index, y=res['NDX_Rebased'], name="Nasdaq 100 (Buy & Hold)",
                       line=dict(color='gray', dash='dot')),
            secondary_y=False, row=1, col=1
        )

        # Trace 3: SMA (Secondary Axis)
        fig.add_trace(
            go.Scatter(x=res.index, y=res[sma_col], name=f"NDX {ma_days}-day SMA",
                       line=dict(color='orange', width=1)),
            secondary_y=True, row=1, col=1
        )

        # Trace 4: NDX Price (Secondary Axis) - for context
        fig.add_trace(
            go.Scatter(x=res.index, y=res['NDX'], name="NDX Price",
                       line=dict(color='cyan', width=1, dash='dot'), opacity=0.5),
            secondary_y=True, row=1, col=1
        )

        # --- Table Calculation ---
        monthly_returns = res['Strategy_Ret'].resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_returns_df = pd.DataFrame({
            'Year': monthly_returns.index.year,
            'Month': monthly_returns.index.month,
            'Return': monthly_returns.values
        })
        returns_pivot = monthly_returns_df.pivot(index='Month', columns='Year', values='Return')
        
        # Calculate Annual Returns
        annual_returns = res['Strategy_Ret'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
        annual_return_row = pd.DataFrame(annual_returns.values.reshape(1, -1), index=['Annual'], columns=returns_pivot.columns)
        
        # Combine and keep numeric values for coloring
        returns_pivot_numeric = pd.concat([returns_pivot, annual_return_row])

        # Format numeric returns to string for display
        returns_pivot = returns_pivot_numeric.applymap(lambda x: f'{x:.2%}' if pd.notna(x) else '')
        
        month_map = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December', 'Annual': 'Annual'}
        returns_pivot.index = returns_pivot.index.map(month_map)
        
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December', 'Annual']
        returns_pivot = returns_pivot.reindex(month_order)


        # Trace 5: Monthly Returns Table
        fig.add_trace(
            go.Table(
                header=dict(values=['Month'] + list(returns_pivot.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[returns_pivot.index] + [returns_pivot[col] for col in returns_pivot.columns],
                           fill_color='white',
                           align='left')
            ),
            row=2, col=1
        )


        # --- Title Generation ---
        title_line1 = f'Growth Strategy: {strategy_underlying} with {ma_days}-SMA'
        if use_vix_filter:
            title_line1 += f' & VIX < {vix_threshold} Filter'
        title_line1 += f' ({backtest_start_date}-Present)'

        title_line2 = f"Strategy CAGR: {cagr_strategy:.2%}, Max DD: {max_dd_strategy:.2%} | B&H CAGR: {cagr_ndx:.2%}, Max DD: {max_dd_ndx:.2%}"

        fig.update_layout(
            title_text=f"{title_line1}<br><sup>{title_line2}</sup>",
            yaxis_title='Portfolio Value ($)',
            yaxis2_title='NDX Price',
            yaxis_type="log",
            yaxis2_type="log",
            template="plotly_dark",
            hovermode="x unified",

            # Set default font color to black for dropdown items, then override for other elements
            font_color="black",
            title_font_color="white",
            legend_font_color="white",
            xaxis_title_font_color="white",
            yaxis_title_font_color="white",
            yaxis2_title_font_color="white",
            xaxis_tickfont_color="white",
            yaxis_tickfont_color="white",
            yaxis2_tickfont_color="white",

            legend=dict(
                x=0.01,
                y=0.99,
                xanchor='left',
                yanchor='top'
            )
        )
        fig.update_annotations(font_color='white')
        fig.update_xaxes(tickfont=dict(color='white'), row=1, col=1)
        fig.update_xaxes(tickformat="%Y", row=2, col=1)
        fig.update_yaxes(tickfont=dict(color='white'), row=2, col=1)
        fig.show()


if __name__ == "__main__":
    run_advanced_backtest(backtest_start_date='2012-01-01', use_vix_filter=True, display_chart=True, vix_threshold=35.0)