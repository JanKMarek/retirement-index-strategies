from retirement_index_strategies.Index_Filters_Leverage import run_advanced_backtest

def main():
    """
    main entry point
    """
    print("Running backtest with 50d moving average for QLD")
    run_advanced_backtest(
        backtest_start_date='2012-01-01',
        ma_days=50,
        use_vix_filter=False,
        display_chart=False,
        strategy_underlying='QLD')

    print("Running backtest with 200d moving average for QLD")
    run_advanced_backtest(
        backtest_start_date='2012-01-01',
        ma_days=200,
        use_vix_filter=False,
        display_chart=False,
        strategy_underlying='QLD')


if __name__ == "__main__":
    main()
