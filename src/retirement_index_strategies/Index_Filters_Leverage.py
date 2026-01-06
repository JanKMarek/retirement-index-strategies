from typing import Any

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors as pcolors
from plotly.subplots import make_subplots
from datetime import date
import dash
from dash import dcc, html, Input, Output, State, dash_table

MONTH_MAP = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August',
             9: 'September', 10: 'October', 11: 'November', 12: 'December', 'Annual': 'Annual'}


def run_backtest(
        backtest_start_date='2000-01-01',
        backtest_end_date=None,
        use_ma_filter=True,
        ma_index='^NDX',
        ma_days=200,  # stay long when index is above this MA average
        use_vix_filter=True,
        vix_threshold=30.0,  # only go long if VIX is above this threshold
        trading_instrument='QLD',
        display_chart='plotly'  # can be False (no chart), 'plotly', or 'dash'
):
    # --- Strategy Parameters Generation ---
    args = locals()
    excluded_keys = ['backtest_start_date', 'backtest_end_date', 'display_chart']
    strategy_params_dict = {k: v for k, v in args.items() if k not in excluded_keys}
    strategy_parameters = str(strategy_params_dict)

    if backtest_end_date is None:
        backtest_end_date = date.today().strftime('%Y-%m-%d')

    cagr_strategy, long_entries, ma_index_name, max_dd_strategy, max_dd_date, res, sma_col = execute_strategy(
        backtest_end_date, backtest_start_date, ma_days, ma_index, trading_instrument, use_ma_filter, use_vix_filter,
        vix_threshold)

    if display_chart == 'plotly':
        display_with_plotly(backtest_end_date, backtest_start_date, cagr_strategy, ma_days,
                            ma_index_name, max_dd_strategy, res, sma_col, trading_instrument,
                            use_ma_filter, use_vix_filter, vix_threshold)
    elif display_chart == 'dash':
        display_with_dash(backtest_end_date, backtest_start_date, cagr_strategy, ma_days,
                          ma_index_name, max_dd_strategy, res, sma_col, trading_instrument,
                          use_ma_filter, use_vix_filter, vix_threshold)

    return cagr_strategy, max_dd_strategy, max_dd_date, strategy_parameters, long_entries


def execute_strategy(backtest_end_date: str | Any, backtest_start_date: str, ma_days: int, ma_index: str,
                     trading_instrument: str, use_ma_filter: bool, use_vix_filter: bool, vix_threshold: float):
    # 1. Data Fetching
    start_date = (pd.to_datetime(backtest_start_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
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

    # Add OHLC data for ma_index for charting
    if isinstance(data.columns, pd.MultiIndex):
        df[f'{ma_index_name}_Open'] = data['Open'][ma_index]
        df[f'{ma_index_name}_High'] = data['High'][ma_index]
        df[f'{ma_index_name}_Low'] = data['Low'][ma_index]
        df[f'{ma_index_name}_Close'] = data['Close'][ma_index]
    else:
        # This case is unlikely but good to handle. Assumes `data` is for `ma_index`.
        df[f'{ma_index_name}_Open'] = data['Open']
        df[f'{ma_index_name}_High'] = data['High']
        df[f'{ma_index_name}_Low'] = data['Low']
        df[f'{ma_index_name}_Close'] = data['Close']

    df = df.dropna(subset=[ma_index_name, 'NDX'])  # Ensure index data exists

    # 2. Indicator Calculation
    # Use the ma_days parameter for the moving average
    sma_col = f'SMA{ma_days}'
    df[sma_col] = df[ma_index_name].rolling(window=ma_days).mean()

    # Fill VIX gaps (VIX data can be spotty in very early years, though robust post-2000)
    df['VIX'] = df['VIX'].ffill()

    # 3. Signal Generation
    # Start with a signal that is always long (True), then apply filters.
    signal_condition = pd.Series(True, index=df.index)

    if use_ma_filter:
        signal_condition = signal_condition & (df[ma_index_name] > df[sma_col])

    if use_vix_filter:
        signal_condition = signal_condition & (df['VIX'] < vix_threshold)

    # If no filters are active, signal will be all 1s (always long).
    df['Signal'] = signal_condition.astype(int).shift(1)  # Shift to prevent lookahead bias

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
    max_dd_date = res['Drawdown_Strategy'].idxmin()

    return cagr_strategy, long_entries, ma_index_name, max_dd_strategy, max_dd_date, res, sma_col


def display_with_plotly(backtest_end_date: str | Any, backtest_start_date: str, cagr_strategy: str,
                        ma_days: int, ma_index_name: float | int | Any,
                        max_dd_strategy: float | int | Any, res, sma_col, trading_instrument: str, use_ma_filter: bool,
                        use_vix_filter: bool, vix_threshold: float):
    fig = create_plotly_figure(backtest_end_date, backtest_start_date, cagr_strategy, ma_days,
                                 ma_index_name, max_dd_strategy, res, sma_col, trading_instrument,
                                 use_ma_filter, use_vix_filter, vix_threshold)
    fig.show()


def create_plotly_figure(backtest_end_date: str | Any, backtest_start_date: str, cagr_strategy: str,
                           ma_days: int, ma_index_name: float | int | Any,
                           max_dd_strategy: float | int | Any, res, sma_col, trading_instrument: str, use_ma_filter: bool,
                           use_vix_filter: bool, vix_threshold: float):

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.5, 0.5],
        specs=[[{"secondary_y": True}], [{"type": "table"}]],
        subplot_titles=("", "Strategy Monthly Returns")
    )

    add_chart(fig, res,
                 backtest_start_date, backtest_end_date, use_ma_filter, use_vix_filter, vix_threshold,
                 sma_col, ma_index_name, ma_days,
                 cagr_strategy, max_dd_strategy, trading_instrument)

    returns_pivot_numeric = get_returns_pivot_df(res)

    # Format for plotly table
    returns_pivot_str = returns_pivot_numeric.copy()
    year_columns = [col for col in returns_pivot_str.columns if col != 'Month']
    for col in year_columns:
        returns_pivot_str[col] = returns_pivot_str[col].apply(lambda x: f'{x:.2%}' if pd.notna(x) else '')

    # Trace 4: Monthly Returns Table
    fig.add_trace(
        go.Table(
            header=dict(values=list(returns_pivot_str.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[returns_pivot_str[col] for col in returns_pivot_str.columns],
                       fill_color='white',
                       align='left')
        ),
        row=2, col=1
    )

    fig.update_xaxes(tickformat="%Y", row=2, col=1)
    fig.update_yaxes(tickfont=dict(color='white'), row=2, col=1)

    return fig


def add_chart(fig, res,
                 backtest_start_date, backtest_end_date, use_ma_filter, use_vix_filter, vix_threshold,
                 sma_col, ma_index_name, ma_days,
                 cagr_strategy, max_dd_strategy, trading_instrument):

    # Trace 1: Portfolio Equity (Log Scale recommended for growth curves)
    fig.add_trace(
        go.Scatter(x=res.index, y=res['Equity'], name=f"Strategy ({trading_instrument} + Filters)",
                   line=dict(color='green', width=2), hoverinfo='none'),
        secondary_y=False, row=1, col=1
    )

    # Trace 2: SMA (Secondary Axis)
    fig.add_trace(
        go.Scatter(x=res.index, y=res[sma_col], name=f"{ma_index_name} {ma_days}-day SMA",
                   line=dict(color='red', width=1), hoverinfo='none'),
        secondary_y=True, row=1, col=1
    )

    # Trace 3: Index Price (Secondary Axis) - for context
    fig.add_trace(
        go.Ohlc(x=res.index,
                open=res[f'{ma_index_name}_Open'],
                high=res[f'{ma_index_name}_High'],
                low=res[f'{ma_index_name}_Low'],
                close=res[f'{ma_index_name}_Close'],
                name=f"{ma_index_name} Price",
                opacity=0.5, hoverinfo='none'),
        secondary_y=True, row=1, col=1
    )

    # --- Title Generation ---
    title_line1 = f'Growth Strategy: {trading_instrument}'
    if use_ma_filter:
        title_line1 += f' with {ma_index_name} {ma_days}-SMA'
    if use_vix_filter:
        title_line1 += f' & VIX < {vix_threshold} Filter'
    title_line1 += f' ({backtest_start_date}-{backtest_end_date})'

    title_line2 = f"Strategy CAGR: {cagr_strategy:.2%}, Max DD: {max_dd_strategy:.2%}"

    fig.update_layout(
        title_text=f"{title_line1}<br><sup>{title_line2}</sup>",
        yaxis_title='Portfolio Value ($)',
        yaxis2_title=f'{ma_index_name} Price',
        yaxis_type="log",
        yaxis2_type="log",
        template="plotly_white",
        hovermode="x",
        margin=dict(l=20, r=20, t=50, b=20),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Rockwell"
        ),

        # Set default font color to black for dropdown items, then override for other elements
        font_color="black",
        title_font_color="black",
        legend_font_color="black",
        xaxis_title_font_color="black",
        yaxis_title_font_color="black",
        yaxis2_title_font_color="black",
        xaxis_tickfont_color="black",
        yaxis_tickfont_color="black",
        yaxis2_tickfont_color="black",

        legend=dict(
            x=0.01,
            y=0.99,
            xanchor='left',
            yanchor='top'
        ),
        xaxis_rangeslider_visible=False
    )
    fig.update_annotations(font_color='black')

    # --- X-axis Ticks Generation ---
    major_tickvals = []
    major_ticktext = []

    # Monthly ticks
    monthly_ticks = pd.date_range(res.index.min().to_period('M').to_timestamp(), res.index.max(), freq='MS')
    for tick in monthly_ticks:
        if tick.month == 1:
            major_tickvals.append(tick)
            major_ticktext.append(tick.strftime('%Y'))
        else:
            # Add minor ticks as annotations
            fig.add_annotation(
                x=tick,
                y=0,
                yref="paper",
                showarrow=False,
                text=tick.strftime('%b')[0],
                textangle=0,
                xanchor='center',
                yanchor='top',
                yshift=-10,  # Adjust this value for position
                font=dict(color="black", size=10)
            )

    fig.update_xaxes(
        tickfont=dict(color='black'),
        row=1, col=1,
        tickvals=major_tickvals,
        ticktext=major_ticktext,
        spikemode='across',
        spikesnap='cursor',
        spikethickness=1,
        spikecolor='black'
    )
    return fig


def get_returns_pivot_df(res):
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
    annual_return_row = pd.DataFrame(annual_returns.values.reshape(1, -1), index=['Annual'],
                                     columns=returns_pivot.columns)

    # Combine and keep numeric values for coloring
    returns_pivot = pd.concat([returns_pivot, annual_return_row])

    returns_pivot.index = returns_pivot.index.map(MONTH_MAP)

    month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
                   'November', 'December', 'Annual']
    returns_pivot = returns_pivot.reindex(month_order)

    # For dash_table, we need to reset index
    returns_pivot = returns_pivot.reset_index().rename(columns={'index': 'Month'})

    return returns_pivot


def display_with_dash(backtest_end_date: str | Any, backtest_start_date: str, cagr_strategy: str,
                      ma_days: int, ma_index_name: float | int | Any,
                      max_dd_strategy: float | int | Any, res, sma_col, trading_instrument: str, use_ma_filter: bool,
                      use_vix_filter: bool, vix_threshold: float):

    chart_fig = create_dash_figure(backtest_end_date, backtest_start_date, cagr_strategy, ma_days,
                                   ma_index_name, max_dd_strategy, res, sma_col, trading_instrument,
                                   use_ma_filter, use_vix_filter, vix_threshold)

    returns_pivot_df = get_returns_pivot_df(res)

    app = dash.Dash(__name__)

    style_data_conditional = []
    # returns_pivot_df can have non-numeric 'Month' column.
    year_columns = [c for c in returns_pivot_df.columns if c != 'Month']

    for col in year_columns:
        style_data_conditional.extend([
            {
                'if': {'filter_query': f'{{{col}}} > 0.05', 'column_id': str(col)},
                'backgroundColor': 'lightgreen',
                'color': 'black'
            },
            {
                'if': {'filter_query': f'{{{col}}} < -0.05', 'column_id': str(col)},
                'backgroundColor': 'lightcoral',
                'color': 'black'
            }
        ])

    columns = [{'name': 'Month', 'id': 'Month', 'type': 'text'}]
    columns.extend([
        {
            'name': str(col),
            'id': str(col),
            'type': 'numeric',
            'format': {'specifier': '.2%'}
        } for col in year_columns
    ])

    app.layout = html.Div([
        dcc.Graph(id='main-chart', figure=chart_fig, style={'height': '50vh'}),
        html.H4("Strategy Monthly Returns", style={'color': 'black'}),
        dash_table.DataTable(
            id='returns-table',
            columns=columns,
            data=returns_pivot_df.to_dict('records'),
            style_cell={'textAlign': 'left', 'backgroundColor': 'white', 'color': 'black'},
            style_header={
                'backgroundColor': 'rgb(30, 30, 30)',
                'fontWeight': 'bold',
                'color': 'white'
            },
            style_data_conditional=style_data_conditional
        )
    ])

    @app.callback(
        Output('main-chart', 'figure'),
        Output('returns-table', 'style_data_conditional'),
        Input('returns-table', 'active_cell'),
        State('main-chart', 'figure'),
        State('returns-table', 'data'),
        State('returns-table', 'columns')
    )
    def update_on_cell_click(active_cell, current_figure, table_data, table_columns):
        # Recreate base style
        year_columns = [c['id'] for c in table_columns if c['id'] != 'Month']
        style_data_conditional = []
        for col in year_columns:
            style_data_conditional.extend([
                {
                    'if': {'filter_query': f'{{{col}}} > 0.05', 'column_id': str(col)},
                    'backgroundColor': 'lightgreen',
                    'color': 'black'
                },
                {
                    'if': {'filter_query': f'{{{col}}} < -0.05', 'column_id': str(col)},
                    'backgroundColor': 'lightcoral',
                    'color': 'black'
                }
            ])

        if not active_cell:
            return current_figure, style_data_conditional

        row = active_cell['row']
        col_id = active_cell['column_id']
        month_name = table_data[row]['Month']

        # Style for highlighted cell
        if col_id != 'Month':
            style_data_conditional.append({
                'if': {'row_index': row, 'column_id': col_id},
                'backgroundColor': 'yellow',
                'color': 'black'
            })

        # Update chart with vertical lines
        if col_id == 'Month' or month_name == 'Annual':
            if 'shapes' in current_figure['layout']:
                current_figure['layout']['shapes'] = []
            return current_figure, style_data_conditional

        try:
            year = int(col_id)
            month_map_inv = {v: k for k, v in MONTH_MAP.items() if isinstance(k, int)}
            month = month_map_inv[month_name]

            start_of_month = f"{year}-{month:02d}-01"
            end_of_month = (pd.to_datetime(start_of_month) + pd.offsets.MonthEnd(1)).strftime('%Y-%m-%d')

            shapes = [
                {
                    'type': 'line', 'xref': 'x', 'yref': 'paper',
                    'x0': start_of_month, 'y0': 0, 'x1': start_of_month, 'y1': 1,
                    'line': {'color': 'RoyalBlue', 'width': 2, 'dash': 'dot'}
                },
                {
                    'type': 'line', 'xref': 'x', 'yref': 'paper',
                    'x0': end_of_month, 'y0': 0, 'x1': end_of_month, 'y1': 1,
                    'line': {'color': 'RoyalBlue', 'width': 2, 'dash': 'dot'}
                }
            ]
            current_figure['layout']['shapes'] = shapes
        except (ValueError, KeyError):
            if 'shapes' in current_figure['layout']:
                current_figure['layout']['shapes'] = []


        return current_figure, style_data_conditional

    app.run(debug=True)


def create_dash_figure(backtest_end_date: str | Any, backtest_start_date: str, cagr_strategy: str,
                       ma_days: int, ma_index_name: float | int | Any,
                       max_dd_strategy: float | int | Any, res, sma_col, trading_instrument: str, use_ma_filter: bool,
                       use_vix_filter: bool, vix_threshold: float):

    fig = make_subplots(
        rows=1, cols=1,
        specs=[[{"secondary_y": True}]],
    )

    add_chart(fig, res,
              backtest_start_date, backtest_end_date, use_ma_filter, use_vix_filter, vix_threshold,
              sma_col, ma_index_name, ma_days,
              cagr_strategy, max_dd_strategy, trading_instrument)

    return fig


if __name__ == "__main__":
    # optimal config
    cagr, max_dd, max_dd_date, params, long_entries = run_backtest(
        backtest_start_date='1999-01-01',
        backtest_end_date=None,  # to present
        ma_index='^NDX',
        use_ma_filter=True,
        ma_days=200,
        use_vix_filter=False,
        vix_threshold=30.0,
        trading_instrument='TQQQ',
        display_chart='dash'
    )
    print("\n--- Function Return Values ---")
    print(f"Strategy Parameters: {params}")
    print(f"Returned CAGR: {cagr:.2%}")
    print(f"Returned Max Drawdown: {max_dd:.2%}")
    print(f"Max Drawdown Date: {max_dd_date.strftime('%Y-%m-%d')}")
    print(f"Long Entries: {long_entries}")
