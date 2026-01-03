import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots
from datetime import date


def run_advanced_backtest(
        backtest_start_date='2000-01-01',
        backtest_end_date=None,
        ma_index='^NDX',
        use_ma_filter=True,
        ma_days=200, # stay long when index is above this MA average
        use_vix_filter=True,
        vix_threshold=30.0, # only go long if VIX is above this threshold
        trading_instrument='QLD',
        display_chart=True
        ):

    # --- Strategy Parameters Generation ---
    args = locals()
    excluded_keys = ['backtest_start_date', 'backtest_end_date', 'display_chart']
    strategy_params_dict = {k: v for k, v in args.items() if k not in excluded_keys}
    strategy_parameters = str(strategy_params_dict)

    if backtest_end_date is None:
        backtest_end_date = date.today().strftime('%Y-%m-%d')

    # 1. Data Fetching (25 Years)
    start_date = '1999-01-01'
    # print("Fetching data...")
    # ^NDX is needed for synthetic returns, ^VIX for volatility filter
    tickers = list(set(['^NDX', '^VIX', trading_instrument, ma_index]))
    data = yf.download(tickers, start=start_date, progress=False)

    # Data Cleaning: Handle MultiIndex if present
    if isinstance(data.columns, pd.MultiIndex):
        df = data.xs('Close', level=0, axis=1) if 'Close' in data.columns.levels[0] else data['Close']
    else:
        df = data['Close']

    ma_index_name = ma_index.replace('^', '')
    # Rename columns for clarity
    rename_map = {'^NDX': 'NDX', '^VIX': 'VIX'}
    rename_map[ma_index] = ma_index_name
    df = df.rename(columns=rename_map)
    df = df.dropna(subset=[ma_index_name, 'NDX'])  # Ensure index data exists

    # 2. Indicator Calculation
    # Use the ma_days parameter for the moving average
    sma_col = f'SMA{ma_days}'
    df[sma_col] = df[ma_index_name].rolling(window=ma_days).mean()

    # Fill VIX gaps (VIX data can be spotty in very early years, though robust post-2000)
    df['VIX'] = df['VIX'].ffill()

    # 3. Signal Generation
    # Start with a signal that is always long (1)
    df['Signal'] = 1
    
    # Signal is true when NDX is above its SMA
    if use_ma_filter:
        signal_condition = (df[ma_index_name] > df[sma_col])
    if use_vix_filter:
        # If the VIX filter is used, the signal is only true if VIX is also below the threshold
        signal_condition &= (df['VIX'] < vix_threshold)
    df['Signal'] = signal_condition.astype(int).shift(1) # Shift to prevent lookahead bias


    # 4. Return Calculation
    df['NDX_Ret'] = df['NDX'].pct_change()
    df[f'{ma_index_name}_Ret'] = df[ma_index_name].pct_change()
    df[f'{trading_instrument}_Real_Ret'] = df[trading_instrument].pct_change()

    # Synthetic Return (e.g., for QLD Pre-2006): 2x NDX Return - approx daily expense ratio (0.95%/252)
    daily_expense = 0.0095 / 252
    df[f'{trading_instrument}_Synth_Ret'] = (2 * df['NDX_Ret']) - daily_expense

    # Combine Real and Synthetic returns
    df['Asset_Ret'] = df[f'{trading_instrument}_Real_Ret'].combine_first(df[f'{trading_instrument}_Synth_Ret'])

    # Strategy Return: Signal (0 or 1) * Asset Return
    df['Strategy_Ret'] = df['Signal'] * df['Asset_Ret']

    # Filter for analysis period
    res = df.loc[backtest_start_date:backtest_end_date].copy()

    # 5. Performance Metrics
    initial_capital = 100000
    n_years = (res.index[-1] - res.index[0]).days / 365.25

    # Count long entries
    long_entries = (res['Signal'].diff() == 1).sum()

    # Strategy Metrics
    res['Equity'] = initial_capital * (1 + res['Strategy_Ret']).cumprod()
    cagr_strategy = (res['Equity'].iloc[-1] / initial_capital) ** (1 / n_years) - 1
    res['Peak_Strategy'] = res['Equity'].cummax()
    res['Drawdown_Strategy'] = (res['Equity'] - res['Peak_Strategy']) / res['Peak_Strategy']
    max_dd_strategy = res['Drawdown_Strategy'].min()
    max_dd_date_strategy = res['Drawdown_Strategy'].idxmin()

    # Benchmark (Buy & Hold) Metrics
    res[f'{ma_index_name}_Rebased'] = initial_capital * (1 + res[f'{ma_index_name}_Ret']).cumprod()
    cagr_benchmark = (res[f'{ma_index_name}_Rebased'].iloc[-1] / initial_capital) ** (1 / n_years) - 1
    res[f'Peak_{ma_index_name}'] = res[f'{ma_index_name}_Rebased'].cummax()
    res[f'Drawdown_{ma_index_name}'] = (res[f'{ma_index_name}_Rebased'] - res[f'Peak_{ma_index_name}']) / res[f'Peak_{ma_index_name}']
    max_dd_benchmark = res[f'Drawdown_{ma_index_name}'].min()
    max_dd_date_benchmark = res[f'Drawdown_{ma_index_name}'].idxmin()

    # 6. Reporting
    # print(f"--- Backtest Report ({backtest_start_date}-{backtest_end_date}) ---")
    # print(f"\n--- Strategy ({trading_instrument} + Filters) ---")
    # print(f"Final Portfolio Value: ${res['Equity'].iloc[-1]:,.0f}")
    # print(f"CAGR: {cagr_strategy:.2%}")
    # print(f"Max Drawdown: {max_dd_strategy:.2%} (Occurred: {max_dd_date_strategy.date()})")
    #
    # print(f"\n--- {ma_index_name} (Buy & Hold) ---")
    # print(f"Final Portfolio Value: ${res[f'{ma_index_name}_Rebased'].iloc[-1]:,.0f}")
    # print(f"CAGR: {cagr_benchmark:.2%}")
    # print(f"Max Drawdown: {max_dd_benchmark:.2%} (Occurred: {max_dd_date_benchmark.date()})")


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
            go.Scatter(x=res.index, y=res['Equity'], name=f"Strategy ({trading_instrument} + Filters)",
                       line=dict(color='green', width=2)),
            secondary_y=False, row=1, col=1
        )

        # Trace 2: Benchmark Index
        fig.add_trace(
            go.Scatter(x=res.index, y=res[f'{ma_index_name}_Rebased'], name=f"{ma_index_name} (Buy & Hold)",
                       line=dict(color='gray', dash='dot')),
            secondary_y=False, row=1, col=1
        )

        # Trace 3: SMA (Secondary Axis)
        fig.add_trace(
            go.Scatter(x=res.index, y=res[sma_col], name=f"{ma_index_name} {ma_days}-day SMA",
                       line=dict(color='orange', width=1)),
            secondary_y=True, row=1, col=1
        )

        # Trace 4: Index Price (Secondary Axis) - for context
        fig.add_trace(
            go.Scatter(x=res.index, y=res[ma_index_name], name=f"{ma_index_name} Price",
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
        title_line1 = f'Growth Strategy: {trading_instrument}'
        if use_ma_filter:
            title_line1 += f' with {ma_index_name} {ma_days}-SMA'
        if use_vix_filter:
            title_line1 += f' & VIX < {vix_threshold} Filter'
        title_line1 += f' ({backtest_start_date}-{backtest_end_date})'

        title_line2 = f"Strategy CAGR: {cagr_strategy:.2%}, Max DD: {max_dd_strategy:.2%} | B&H CAGR: {cagr_benchmark:.2%}, Max DD: {max_dd_benchmark:.2%}"

        fig.update_layout(
            title_text=f"{title_line1}<br><sup>{title_line2}</sup>",
            yaxis_title='Portfolio Value ($)',
            yaxis2_title=f'{ma_index_name} Price',
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
    
    return cagr_strategy, max_dd_strategy, strategy_parameters, long_entries

if __name__ == "__main__":
    # optimal config
    cagr, max_dd, params, long_entries = run_advanced_backtest(
        backtest_start_date='2012-01-01',
        backtest_end_date=None, # to present
        ma_index='^NDX',
        use_ma_filter=True,
        ma_days=200,
        use_vix_filter=False,
        vix_threshold=30.0,
        trading_instrument='QQQ',
        display_chart=True
    )
    print("\n--- Function Return Values ---")
    print(f"Strategy Parameters: {params}")
    print(f"Returned CAGR: {cagr:.2%}")
    print(f"Returned Max Drawdown: {max_dd:.2%}")
    print(f"Long Entries: {long_entries}")
