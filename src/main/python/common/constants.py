# Columns when extracted from yahoo finance

HIGH = 'High'
LOW = 'Low'
OPEN = 'Open'
CLOSE = 'Close'
VOLUME = 'Volume'
ADJ_CLOSE = 'Adj Close'
TICKER = 'Ticker'

DATE_STR_FORMAT = '%Y-%m-%d'

DATE = 'Date'

LOG_RET_DLY = 'log_ret_dly'
ADJ_CLOSE_PRC = 'Adj Close'
VOLATILITY = 'volatility'

NUM_TRADE_DAYS_PER_YR = 252

# for the backtesting,py file
NUM_TRADE_DAYS_PER_MONTH = 21
COL_NUM_TRADE_DAYS_YR_TREND = '{}_days'.format(NUM_TRADE_DAYS_PER_YR)
COL_NUM_TRADE_DAYS_MONTH_TREND = '{}_days'.format(NUM_TRADE_DAYS_PER_MONTH)
BUY = 'BUY'
SELL = 'SELL'
HOLD = 'HOLD'

SIGNAL_VAL = 'signal_val'
SIGNAL_NAME = 'signal_name'
SIGNALS_DICT = {1: BUY, -1: SELL, 0: HOLD}

RESIDUAL_RETURN = 'residual_return'
RESIDUAL_RISK = 'residual_risk'
EXP_RESIDUAL_RETURN = 'expected_residual_return'
EXCESS_RETURN = 'excess_return'