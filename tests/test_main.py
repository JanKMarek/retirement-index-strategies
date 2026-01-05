from retirement_index_strategies import main
import pandas as pd
from retirement_index_strategies.Index_Filters_Leverage import run_backtest


def test_buy_and_hold(mocker):
    mocker.patch('retirement_index_strategies.Index_Filters_Leverage.run_backtest',
                 return_value=(0.2, 0.3, pd.to_datetime('2022-01-01'), "{'param': 'value'}", 10))
    main.buy_and_hold()
    # Add assertions here to check the output of the function


def test_find_optimal_configuration(mocker):
    mocker.patch('retirement_index_strategies.Index_Filters_Leverage.run_backtest',
                 return_value=(0.2, 0.3, pd.to_datetime('2022-01-01'), "{'param': 'value'}", 10))
    main.find_optimal_configuration()
    # Add assertions here to check the output of the function


def test_run_backtest_dash_display(mocker):
    mocker.patch('dash.Dash.run_server')
    run_backtest(display_chart='dash')
    # Add assertions here to check if the dash server was called
