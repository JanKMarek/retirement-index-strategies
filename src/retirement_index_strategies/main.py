from retirement_index_strategies.Index_Filters_Leverage import run_advanced_backtest

def buy_and_hold():

    config_params = {
        'backtest_start_date': '2007-01-01',
        'use_ma_filter': False,
        'use_vix_filter': False,
        'trading_instrument': 'QQQ'
    }

    cagr, max_dd, strategy_params, long_entries = run_advanced_backtest(**config_params)
    print(f"{cagr:.0%}, {max_dd:.0%}, {long_entries}, {strategy_params}")


def find_optimal_configuration():

    config_params = [
        {'ma_days': 200, 'use_vix_filter': True, 'vix_threshold': 30.0, 'trading_instrument': 'QQQ'},
        {'ma_days': 200, 'use_vix_filter': False, 'vix_threshold': 30.0, 'trading_instrument': 'QQQ'},
        {'ma_days': 50, 'use_vix_filter': True, 'vix_threshold': 30.0, 'trading_instrument': 'QQQ'},
        {'ma_days': 50, 'use_vix_filter': False, 'vix_threshold': 30.0, 'trading_instrument': 'QQQ'},

        {'ma_days': 200, 'use_vix_filter': True, 'vix_threshold': 30.0, 'trading_instrument': 'QLD'},
        {'ma_days': 200, 'use_vix_filter': False, 'vix_threshold': 30.0, 'trading_instrument': 'QLD'},
        {'ma_days': 50, 'use_vix_filter': True, 'vix_threshold': 30.0, 'trading_instrument': 'QLD'},
        {'ma_days': 50, 'use_vix_filter': False, 'vix_threshold': 30.0, 'trading_instrument': 'QLD'},

        {'ma_days': 200, 'use_vix_filter': True, 'vix_threshold': 30.0, 'trading_instrument': 'TQQQ'},
        {'ma_days': 200, 'use_vix_filter': False, 'vix_threshold': 30.0, 'trading_instrument': 'TQQQ'},
        {'ma_days': 50, 'use_vix_filter': True, 'vix_threshold': 30.0, 'trading_instrument': 'TQQQ'},
        {'ma_days': 50, 'use_vix_filter': False, 'vix_threshold': 30.0, 'trading_instrument': 'TQQQ'},
    ]

    for element in config_params:
        element['backtest_start_date'] = '2007-01-01'
        element['display_chart'] = False

    print(f"CAGR, Max Drawdown, Long Entries, Strategy Parameters")
    for params_set in config_params:
        if (not params_set['use_vix_filter']) and params_set['ma_days'] == 200:
            cagr, max_dd, strategy_params, long_entries = run_advanced_backtest(**params_set)
            print(f"{cagr:.0%}, {max_dd:.0%}, {long_entries}, {strategy_params}")


if __name__ == "__main__":

    buy_and_hold()

    find_optimal_configuration()
