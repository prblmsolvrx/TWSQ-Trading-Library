### TWSQ Trading Library - Comprehensive Documentation


## Table of Contents
1. [Introduction](#introduction)
2. [Strategy Construction](#strategy-construction)
3. [Backtesting](#backtesting)
4. [Live Trading](#live-trading)

## Introduction

The TWSQ Trading Library is a powerful tool designed for backtesting and live trading of fully automated quantitative strategies in cryptocurrency markets. It specializes in deploying strategies based on price-volume data, which are particularly effective in less efficient markets like cryptocurrencies.

### Key Features:
1. Backtesting capabilities
2. Live trading on Kraken (with plans to add more exchanges)
3. Focus on price-volume based strategies (statistical arbitrage)
4. Extensible for additional datasets and asset classes

### Current Limitations:
- Live trading is currently only available on Kraken
- Primary focus is on price-volume data (though custom data can be incorporated)

### Installation

1. Download the `twsq` codebase
2. Open your terminal or command prompt
3. Navigate to the downloaded folder
4. Run the following command:
   ```
   python setup.py install
   ```

For developers who want to build on top of this library:
```
python setup.py develop
```

This allows any changes made to the code to be immediately reflected without the need for reinstallation.

### Setting up TWSQROOT Directory

The library needs a location to save data and results. By default, it creates a folder called `MyTWSQ` in your home directory.

To set a custom location:
1. Create an environment variable named `TWSQROOT`
2. Set it to your desired path, e.g., `/User1/codebase/`

The library will then create and use a folder `/User1/codebase/MyTWSQ`

To verify your TWSQROOT location:

```python
from twsq.utils import get_twsqroot
get_twsqroot()
```

Data saved in this directory includes:
1. Price-volume cache files
2. Backtest / live trading results (positions, PnL, etc.)
3. Log files

## Strategy Construction

### Creating a Strategy

To create a strategy using the TWSQ Trading Library, follow these steps:

1. Import the `Alpha` class from `twsq.alpha`
2. Create your strategy as a subclass of the `Alpha` class
3. Define key functions within your subclass to specify the strategy's behavior

Here's a simple example of a strategy that buys 1 ETH/USD on each rebalance:

```python
from twsq.alpha import Alpha

class ETHDCA(Alpha):
    def rebalance(self):
        self.create_order(
            'ETH/USD',
            1,
            'buy',
            route=True)
    return
```

### Key Functions

#### 1. rebalance()

This is the most important function to define for each strategy. It gets executed at a certain frequency (e.g., every hour, day) that you set when running the backtester or live trader.

#### 2. prepare()

Use this function to set strategy parameters and prepare for trading. It's called once before the first rebalance.

```python
def prepare(self, param1, param2):
    self.param1 = param1
    self.param2 = param2
```

#### 3. on_finished_order()

This function is automatically called when an order is finished. It's useful for "market-making" or "liquidity-providing" strategies.

```python
def on_finished_order(self, order):
    # React to the finished order
    pass
```

#### 4. on_exit()

This custom function runs when the trader exits. It's useful for cleanup operations.

```python
def on_exit(self):
    self.cancel_all_orders()
```

### Pre-built Functions

The `Alpha` class provides several pre-built functions that you can use in your strategy:

1. `create_order()`: Create a new order
2. `trade_to_target()`: Trade to specified target positions
3. `cancel_all_orders()`: Cancel all open orders
4. `cancel_order()`: Cancel a specific order
5. `get_open_orders()`: Get a list of open orders
6. `get_current_price()`: Get the current price of a symbol
7. `get_lastn_bars()`: Get the last n OHLCV bars of a symbol
8. `get_pos()`: Get your current positions

### Example: Using Pre-built Functions

Here's an example of how to use some of these pre-built functions:

```python
class ExampleStrategy(Alpha):
    def rebalance(self):
        # Get current price
        eth_price = self.get_current_price('ETH/USD')
        
        # Get last 5 1-hour bars
        bars = self.get_lastn_bars('ETH/USD', 5, '1h')
        
        # Make a decision based on the data
        if some_condition:
            # Create a buy order
            self.create_order('ETH/USD', 1, 'buy', route=True)
        else:
            # Cancel all open orders
            self.cancel_all_orders()
        
        # Get current positions
        positions = self.get_pos()
        print(f"Current ETH position: {positions.get('ETH', 0)}")
```

## Backtesting

Backtesting is a crucial feature of the TWSQ Trading Library that allows you to test your strategies on historical data before deploying them in live trading.

### Running a Backtest

To run a backtest, use the `run_backtest` method of your strategy class:

```python
result = ETHDCA.run_backtest(start_ts='20210101')
```

### Backtest Parameters

The `run_backtest` method accepts several parameters:

- `start_ts`: Start time of the backtest (default is 365 days prior to `end_ts`)
- `end_ts`: End time of the backtest (default is current time)
- `freq`: Rebalance frequency (default is '1h')
- `name`: Name of the strategy (used for logging and saving results)
- `taker_fee`: Commissions for market orders (default is Kraken's fee of 26 bps)
- `maker_fee`: Commissions for limit orders (default is Kraken's fee of 16 bps)
- `slip`: Slippage for market orders (default is 10 bps)

Example with custom parameters:

```python
result = ETHDCA.run_backtest(
    start_ts='20210101',
    end_ts='20211231',
    freq='1d',
    name='ETHDCA_2021',
    taker_fee=0.0025,
    maker_fee=0.0015,
    slip=0.001
)
```

### Backtest Logic

The backtest iterates through historical data at the specified frequency:

1. Fill any outstanding limit orders
2. Call the strategy's `rebalance` function
3. Fill any market orders created by `rebalance`
4. Move to the next time step

Market orders are filled immediately at the current price plus slippage. Limit orders are checked for filling at each time step based on the high and low prices of the bar.

### PnL Calculation

The backtest PnL is the aggregate PnL of all trades executed by the strategy. There's no notion of "budget" or "equity capital", so PnL can be infinitely negative in the backtest.

### Shorting

Shorting is currently permitted in backtests. Work is ongoing to implement shorting in live trading as well.

### Output

Backtest results are saved in `MyTWSQ/alphas/{strategy_name}/backtest/` as two CSV files:

1. `pos_pnl.csv`: Contains positions and PnL information
   - Columns: Date, asset positions, port_val, pnl
2. `orders.csv`: Contains all order information
   - Columns: strategy, symbol, qty, side, sec_type, limit_price, type, custom_id, id, status, qty_filled, ntn_filled, avg_px, fee, start_ts, arrival_px, base, quote, end_ts

You can also access these results through the `result` object returned by `run_backtest`:

```python
positions_and_pnl = result.pos_pnl
orders = result.orders
```

### Multiple Backtests

You can run multiple backtests with different parameters to compare results:

```python
net = ETHDCA.run_backtest(start_ts='20210101', name='Net', freq='1d')
gross = ETHDCA.run_backtest(start_ts='20210101', name='Gross', freq='1d', 
                            taker_fee=0, maker_fee=0, slip=0)
```

## Live Trading

The TWSQ Trading Library supports live trading of your strategies on cryptocurrency exchanges. Currently, live trading is implemented for Kraken.

### Setting Up Live Trading

#### 1. Configure Settings File

Create a `settings.yml` file in your `MyTWSQ` folder with your API credentials:

```yaml
exchange_apis:
  Kraken:
    key: your_api_key_here
    secret: your_api_secret_here
```

#### 2. Launch Live Trading

Use the `run_live` method to start live trading:

```python
ETHDCA.run_live()
```

### Live Trading Parameters

The `run_live` method accepts several parameters:

- `freq`: Rebalance frequency (default is '1h')
- `name`: Name of the trading session (used for logging and saving results)

Example with custom parameters:

```python
ETHDCA.run_live(freq='30m', name='ETHDCA_Live_30m')
```

### Rebalance Frequency

You can set the rebalance frequency in two ways:

1. Use preset frequencies: '1m', '5m', '15m', '30m', '1h', '4h', '1d'
   - Rebalancing occurs at bar start times
2. Specify a float value for the number of seconds between rebalances
   - Rebalancing occurs immediately and then every `freq` seconds

### Exchange API Usage

The trader uses a combination of REST and WebSocket APIs:

- WebSocket: Used for real-time price data and order updates
- REST: Used for submitting orders

### Exiting Live Trading

To exit live trading, use a keyboard interruption (CTRL+C). The `on_exit` function will be called before the trader exits.

### Output

Live trading results are saved in `MyTWSQ/alphas/{strategy_name}/live_trading/broker/`:

1. `logs/`: Contains timestamped log files
2. `pos_pnl.csv`: Contains positions and PnL information
3. `orders.csv`: Contains all order information

These files maintain continuity across trading sessions for the same strategy.

### Position Tracking

Strategy positions are maintained across trading sessions. If you restart a trading session with the same strategy name, it will load the previous positions from the `pos_pnl.csv` file.

### Telegram Logging

You can set up Telegram logging to receive mobile notifications about your live trader:

1. Create a Telegram bot using BotFather
2. Obtain your `token` and `chat_id`
3. Add these to your `settings.yml` file:

```yaml
telegram:
    token: your_token_here
    chat_id: your_chat_id_here
```

This will send all logging information to your Telegram bot.

### Considerations for Live Trading

1. Ensure your strategy is well-tested in backtests before live trading
2. Be aware of exchange rate limits for API calls
3. Monitor your strategy regularly, especially in the initial stages
4. Use appropriate risk management techniques
5. Be prepared for potential connectivity issues or exchange downtime

This comprehensive documentation provides a thorough overview of the TWSQ Trading Library, covering its introduction, strategy construction, backtesting, and live trading capabilities. Users can refer to this document for detailed information on how to use and customize the library for their trading needs.