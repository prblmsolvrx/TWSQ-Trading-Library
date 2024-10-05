# TWSQ Trading Library

## Table of Contents

1. [Introduction](#introduction)
2. [Installation and Setup](#installation-and-setup)
3. [Strategy Construction](#strategy-construction)
4. [Backtesting](#backtesting)
5. [Live Trading](#live-trading)
6. [Advanced Features](#advanced-features)
7. [Best Practices and Tips](#best-practices-and-tips)
8. [Troubleshooting](#troubleshooting)
9. [Glossary](#glossary)

## 1. Introduction

The TWSQ Trading Library is a sophisticated tool designed for quantitative traders and researchers in the cryptocurrency markets. It offers a powerful platform for developing, backtesting, and deploying automated trading strategies with a focus on price-volume data analysis.

### 1.1 Key Features

- **Backtesting Engine**: Robust system for testing strategies against historical data
- **Live Trading Capabilities**: Seamless transition from backtesting to live trading on supported exchanges
- **Price-Volume Strategy Focus**: Specialized tools for strategies leveraging price and volume data
- **Extensibility**: Flexible architecture allowing for integration of custom datasets and expansion to other asset classes
- **Performance Optimization**: Efficient data handling and strategy execution for high-frequency trading

### 1.2 Current Limitations

- Live trading is currently limited to the Kraken exchange
- Primary focus on cryptocurrency markets, though extensible to other asset classes
- Emphasis on price-volume based strategies, with ongoing work to incorporate more diverse data sources

### 1.3 Target Users

- Quantitative traders and researchers
- Cryptocurrency market specialists
- Financial institutions exploring algorithmic trading in crypto markets
- Academic researchers studying market microstructure and trading strategies

## 2. Installation and Setup

### 2.1 System Requirements

- Python 3.7 or higher
- 8GB RAM (minimum), 16GB RAM (recommended)
- High-speed internet connection for live trading

### 2.2 Installation Process

1. Download the TWSQ codebase:
   ```
   git clone https://github.com/twsq/trading-library.git
   ```

2. Navigate to the downloaded folder:
   ```
   cd trading-library
   ```

3. Install the library:
   For users:
   ```
   python setup.py install
   ```
   For developers:
   ```
   python setup.py develop
   ```

### 2.3 Setting up TWSQROOT Directory

The TWSQROOT directory is crucial for storing data, results, and configurations. By default, it's created in your home directory as `MyTWSQ`.

To set a custom location:

1. Create an environment variable named `TWSQROOT`
2. Set it to your desired path, e.g., `/User1/codebase/`

Verify your TWSQROOT location:

```python
from twsq.utils import get_twsqroot
print(get_twsqroot())
```

### 2.4 Directory Structure

```
MyTWSQ/
├── alphas/
│   ├── strategy_name1/
│   │   ├── backtest/
│   │   └── live_trading/
│   └── strategy_name2/
├── data/
│   ├── price_volume_cache/
│   └── custom_data/
├── logs/
└── settings.yml
```

### 2.5 Configuration

Create a `settings.yml` file in your TWSQROOT directory:

```yaml
exchange_apis:
  Kraken:
    key: your_api_key_here
    secret: your_api_secret_here

telegram:
  token: your_telegram_bot_token
  chat_id: your_telegram_chat_id

# Add other global settings here
```

## 3. Strategy Construction

### 3.1 The Alpha Class

The `Alpha` class is the foundation for all trading strategies in the TWSQ library. To create a strategy:

1. Import the `Alpha` class
2. Create a subclass of `Alpha`
3. Implement required methods

Example:

```python
from twsq.alpha import Alpha

class MyStrategy(Alpha):
    def prepare(self):
        # Initialize strategy parameters
        pass

    def rebalance(self):
        # Main strategy logic
        pass
```

### 3.2 Key Methods

#### 3.2.1 prepare()

Called once before trading begins. Use it to set up strategy parameters.

```python
def prepare(self):
    self.lookback = 20
    self.threshold = 0.02
```

#### 3.2.2 rebalance()

The core of your strategy. Called at each rebalance interval.

```python
def rebalance(self):
    price = self.get_current_price('BTC/USD')
    if self.should_buy(price):
        self.create_order('BTC/USD', 0.1, 'buy')
```

#### 3.2.3 on_finished_order()

Called when an order is completed. Useful for order-driven strategies.

```python
def on_finished_order(self, order):
    if order.side == 'buy':
        self.create_order(order.symbol, order.filled, 'sell', limit_price=order.avg_price * 1.05)
```

#### 3.2.4 on_exit()

Called when the trading session ends. Use for cleanup operations.

```python
def on_exit(self):
    self.cancel_all_orders()
    self.log_final_stats()
```

### 3.3 Built-in Functions

The `Alpha` class provides numerous utility functions:

- `create_order(symbol, quantity, side, **kwargs)`: Place a new order
- `trade_to_target(targets)`: Adjust positions to match target allocations
- `cancel_all_orders()`: Cancel all open orders
- `get_current_price(symbol)`: Get the latest price for a symbol
- `get_lastn_bars(symbol, n, timeframe)`: Get historical price bars
- `get_pos()`: Get current positions

Example usage:

```python
class TrendFollower(Alpha):
    def rebalance(self):
        symbol = 'ETH/USD'
        bars = self.get_lastn_bars(symbol, 50, '1h')
        sma = bars['close'].mean()
        current_price = self.get_current_price(symbol)

        if current_price > sma * 1.02:
            self.create_order(symbol, 0.1, 'buy')
        elif current_price < sma * 0.98:
            self.create_order(symbol, 0.1, 'sell')
```

### 3.4 Custom Indicators and Signals

You can extend your strategy with custom indicators:

```python
import pandas as pd

class RSIStrategy(Alpha):
    def calculate_rsi(self, symbol, period=14):
        bars = self.get_lastn_bars(symbol, period+1, '1h')
        close_diff = bars['close'].diff()
        gain = close_diff.where(close_diff > 0, 0).rolling(window=period).mean()
        loss = -close_diff.where(close_diff < 0, 0).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def rebalance(self):
        symbol = 'BTC/USD'
        rsi = self.calculate_rsi(symbol)
        if rsi.iloc[-1] < 30:
            self.create_order(symbol, 0.01, 'buy')
        elif rsi.iloc[-1] > 70:
            self.create_order(symbol, 0.01, 'sell')
```

## 4. Backtesting

### 4.1 Running a Backtest

Use the `run_backtest` method to test your strategy:

```python
result = MyStrategy.run_backtest(
    start_ts='20210101',
    end_ts='20211231',
    freq='1h',
    name='MyStrategy_2021'
)
```

### 4.2 Backtest Parameters

- `start_ts`: Start time of the backtest (default: 365 days before `end_ts`)
- `end_ts`: End time of the backtest (default: current time)
- `freq`: Rebalance frequency (default: '1h')
- `name`: Strategy name for logging and results
- `taker_fee`: Fee for market orders (default: 0.0026)
- `maker_fee`: Fee for limit orders (default: 0.0016)
- `slip`: Slippage for market orders (default: 0.0010)

### 4.3 Backtest Logic

1. Initialize strategy and load historical data
2. Iterate through time steps:
   a. Fill any pending limit orders
   b. Call `rebalance()`
   c. Execute any new market orders
   d. Calculate and record positions and PnL
3. Generate and save results

### 4.4 Understanding Results

Backtest results are saved in `MyTWSQ/alphas/{strategy_name}/backtest/`:

1. `pos_pnl.csv`: Positions and PnL over time
2. `orders.csv`: Detailed order information

Access results programmatically:

```python
positions_and_pnl = result.pos_pnl
orders = result.orders

print(f"Final PnL: ${positions_and_pnl['pnl'].iloc[-1]:.2f}")
print(f"Total trades: {len(orders)}")
```

### 4.5 Analyzing Performance

Use the built-in analysis tools:

```python
from twsq.analysis import calculate_sharpe_ratio, plot_equity_curve

sharpe = calculate_sharpe_ratio(result.pos_pnl['pnl'])
print(f"Sharpe Ratio: {sharpe:.2f}")

plot_equity_curve(result.pos_pnl['pnl'])
```

### 4.6 Parameter Optimization

Perform grid search for optimal parameters:

```python
from itertools import product

def run_backtest_with_params(lookback, threshold):
    class OptimizedStrategy(Alpha):
        def prepare(self):
            self.lookback = lookback
            self.threshold = threshold

        def rebalance(self):
            # Strategy logic using self.lookback and self.threshold
            pass

    result = OptimizedStrategy.run_backtest(start_ts='20210101', end_ts='20211231')
    return result.pos_pnl['pnl'].iloc[-1]

lookbacks = [10, 20, 30]
thresholds = [0.01, 0.02, 0.03]

results = []
for lookback, threshold in product(lookbacks, thresholds):
    pnl = run_backtest_with_params(lookback, threshold)
    results.append((lookback, threshold, pnl))

best_params = max(results, key=lambda x: x[2])
print(f"Best parameters: Lookback={best_params[0]}, Threshold={best_params[1]}, PnL=${best_params[2]:.2f}")
```

## 5. Live Trading

### 5.1 Preparing for Live Trading

1. Ensure your strategy is well-tested in backtests
2. Set up API credentials in `settings.yml`
3. Implement proper risk management in your strategy

### 5.2 Starting Live Trading

Use the `run_live` method:

```python
MyStrategy.run_live(freq='1h', name='MyStrategy_Live')
```

### 5.3 Live Trading Parameters

- `freq`: Rebalance frequency (default: '1h')
- `name`: Name for the trading session

### 5.4 Monitoring Live Performance

Live trading results are saved in `MyTWSQ/alphas/{strategy_name}/live_trading/broker/`:

- `logs/`: Timestamped log files
- `pos_pnl.csv`: Real-time positions and PnL
- `orders.csv`: Detailed order information

### 5.5 Risk Management

Implement risk management in your strategy:

```python
class SafeStrategy(Alpha):
    def prepare(self):
        self.max_position = 1.0  # BTC
        self.stop_loss_pct = 0.05

    def rebalance(self):
        current_pos = self.get_pos().get('BTC', 0)
        if current_pos >= self.max_position:
            return  # Don't open new positions

        # Check for stop loss
        entry_price = self.get_average_entry_price('BTC/USD')
        current_price = self.get_current_price('BTC/USD')
        if current_price < entry_price * (1 - self.stop_loss_pct):
            self.close_all_positions()
            return

        # Regular strategy logic
        if self.should_buy():
            self.create_order('BTC/USD', 0.1, 'buy')
```

### 5.6 Handling Errors and Disconnections

Implement error handling and reconnection logic:

```python
class RobustStrategy(Alpha):
    def rebalance(self):
        try:
            # Regular strategy logic
            pass
        except ConnectionError:
            self.log.warning("Connection error. Retrying in 60 seconds.")
            time.sleep(60)
            self.reconnect()
        except Exception as e:
            self.log.error(f"Unexpected error: {str(e)}")
            self.on_exit()
```

## 6. Advanced Features

### 6.1 Custom Data Integration

Integrate external data sources:

```python
import requests

class NewsStrategy(Alpha):
    def prepare(self):
        self.api_key = "your_news_api_key"

    def get_news_sentiment(self, symbol):
        response = requests.get(f"https://cryptonews-api.com/api/v1?tickers={symbol}&items=50&token={self.api_key}")
        news = response.json()['data']
        # Process news and calculate sentiment
        return sentiment_score

    def rebalance(self):
        symbol = 'BTC/USD'
        sentiment = self.get_news_sentiment(symbol)
        if sentiment > 0.5:
            self.create_order(symbol, 0.1, 'buy')
        elif sentiment < -0.5:
            self.create_order(symbol, 0.1, 'sell')
```

### 6.2 Multi-Asset Strategies

Create strategies that trade multiple assets:

```python
class DiversifiedStrategy(Alpha):
    def prepare(self):
        self.assets = ['BTC/USD', 'ETH/USD', 'XRP/USD']
        self.allocation = {asset: 1/len(self.assets) for asset in self.assets}

    def rebalance(self):
        current_prices = {asset: self.get_current_price(asset) for asset in self.assets}
        current_values = {asset: self.get_pos().get(asset.split('/')[0], 0) * price
                          for asset, price in current_prices.items()}

        total_value = sum(current_values.values())
        targets = {asset: self.allocation[asset] * total_value / current_prices[asset]
                   for asset in self.assets}

        self.trade_to_target(targets)
```

### 6.3 Event-Driven Strategies
```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

class EventDrivenStrategy(Alpha):
    def prepare(self):
        self.last_price = None
        self.volatility_threshold = 0.02  # Added a default value

    def on_price_update(self, symbol, price):
        if self.last_price is None:
            self.last_price = price
            return
        price_change = abs(price - self.last_price) / self.last_price
        if price_change > self.volatility_threshold:
            if price > self.last_price:
                self.create_order(symbol, 0.1, 'buy')
            else:
                self.create_order(symbol, 0.1, 'sell')
        self.last_price = price

    def rebalance(self):
        # This method is still called periodically
        # You can use it for other periodic checks or rebalancing
        pass

class MLStrategy(Alpha):
    def prepare(self):
        self.model = RandomForestClassifier(n_estimators=100)
        self.train_model()

    def train_model(self):
        # Fetch historical data
        data = self.get_lastn_bars('BTC/USD', 1000, '1h')
        # Create features (e.g., moving averages, RSI, etc.)
        data['SMA_20'] = data['close'].rolling(window=20).mean()
        data['RSI'] = self.calculate_rsi(data['close'])
        # Create labels (1 for price increase, 0 for decrease)
        data['label'] = (data['close'].shift(-1) > data['close']).astype(int)
        # Prepare features and labels
        features = data[['SMA_20', 'RSI']].dropna()
        labels = data['label'].dropna()
        # Split data and train model
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
        self.model.fit(X_train, y_train)
        # Print model accuracy
        accuracy = self.model.score(X_test, y_test)
        self.log.info(f"Model accuracy: {accuracy:.2f}")

    def rebalance(self):
        current_data = self.get_lastn_bars('BTC/USD', 20, '1h')
        features = np.array([[
            current_data['close'].rolling(window=20).mean().iloc[-1],
            self.calculate_rsi(current_data['close']).iloc[-1]
        ]])
        prediction = self.model.predict(features)[0]
        if prediction == 1:
            self.create_order('BTC/USD', 0.1, 'buy')
        else:
            self.create_order('BTC/USD', 0.1, 'sell')

    def calculate_rsi(self, prices, period=14):
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)

        return rsi
```

### 6.5 Custom Order Types

Implement advanced order types:

```python
class AdvancedOrderStrategy(Alpha):
    def create_trailing_stop_order(self, symbol, quantity, stop_percent):
        current_price = self.get_current_price(symbol)
        stop_price = current_price * (1 - stop_percent)

        order = self.create_order(symbol, quantity, 'sell', type='trailing_stop', stop_price=stop_price)
        self.monitor_trailing_stop(order, stop_percent)

    def monitor_trailing_stop(self, order, stop_percent):
        while order.status != 'filled':
            current_price = self.get_current_price(order.symbol)
            new_stop_price = current_price * (1 - stop_percent)

            if new_stop_price > order.stop_price:
                order.modify(stop_price=new_stop_price)

            time.sleep(60)  # Check every minute

    def rebalance(self):
        if self.should_buy('BTC/USD'):
            self.create_order('BTC/USD', 0.1, 'buy')
            self.create_trailing_stop_order('BTC/USD', 0.1, 0.05)  # 5% trailing stop
```

## 7. Best Practices and Tips

### 7.1 Strategy Development

1. **Start Simple**: Begin with basic strategies and gradually increase complexity.
2. **Modular Design**: Break your strategy into reusable components for easier testing and maintenance.
3. **Version Control**: Use git to track changes in your strategy code.
4. **Documentation**: Thoroughly document your strategy logic, parameters, and assumptions.

### 7.2 Backtesting

1. **Data Quality**: Ensure your historical data is accurate and free from errors.
2. **Realistic Assumptions**: Use realistic fees, slippage, and fill probabilities.
3. **Out-of-Sample Testing**: Test your strategy on data not used during development.
4. **Sensitivity Analysis**: Test your strategy with different parameters to ensure robustness.

### 7.3 Risk Management

1. **Position Sizing**: Implement proper position sizing based on your risk tolerance.
2. **Stop Losses**: Use stop-loss orders to limit potential losses.
3. **Diversification**: Trade multiple assets to spread risk.
4. **Correlation Analysis**: Be aware of correlations between traded assets.

### 7.4 Live Trading

1. **Paper Trading**: Test your strategy in a live environment without real money first.
2. **Monitoring**: Continuously monitor your strategy's performance and market conditions.
3. **Failsafes**: Implement automatic shutoff mechanisms for unexpected scenarios.
4. **Gradual Deployment**: Start with small positions and gradually increase as you gain confidence.

### 7.5 Performance Optimization

1. **Profiling**: Use Python's built-in profiling tools to identify bottlenecks.
2. **Caching**: Cache frequently used data to reduce API calls and computation time.
3. **Vectorization**: Use numpy and pandas for efficient data manipulation.
4. **Asynchronous Programming**: Utilize async functions for I/O-bound operations.

## 8. Troubleshooting

### 8.1 Common Issues and Solutions

1. **API Connection Errors**
    - Check internet connection
    - Verify API credentials in `settings.yml`
    - Ensure you're not exceeding API rate limits
2. **Unexpected Strategy Behavior**
    - Double-check strategy logic and parameters
    - Verify data quality and timeliness
    - Use logging to track decision-making process
3. **Performance Issues**
    - Profile your code to identify bottlenecks
    - Optimize data handling and calculations
    - Consider upgrading hardware for resource-intensive strategies
4. **Backtesting Discrepancies**
    - Ensure backtest parameters match live trading conditions
    - Check for look-ahead bias in your strategy
    - Verify historical data accuracy

### 8.2 Debugging Techniques

1. **Logging**: Implement comprehensive logging in your strategy.

```python
import logging

class DebuggableStrategy(Alpha):
    def prepare(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

    def rebalance(self):
        self.logger.debug(f"Current positions: {self.get_pos()}")
        self.logger.debug(f"Available balance: {self.get_balance()}")

        # Strategy logic here

        self.logger.info(f"Placed order: {order_details}")
```

2. **Dry Runs**: Implement a dry run mode for testing without placing orders.

```python
class DryRunStrategy(Alpha):
    def __init__(self, *args, dry_run=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dry_run = dry_run

    def create_order(self, *args, **kwargs):
        if self.dry_run:
            self.log.info(f"Dry run: Would create order with args: {args}, kwargs: {kwargs}")
        else:
            super().create_order(*args, **kwargs)
```

3. **Unit Testing**: Create unit tests for individual components of your strategy.

```python
import unittest

class TestStrategyComponents(unittest.TestCase):
    def setUp(self):
        self.strategy = MyStrategy()

    def test_rsi_calculation(self):
        test_data = pd.Series([10, 12, 15, 14, 13, 11, 12, 15, 17, 16])
        expected_rsi = 61.90  # Calculated manually or with a trusted source
        calculated_rsi = self.strategy.calculate_rsi(test_data)
        self.assertAlmostEqual(calculated_rsi, expected_rsi, places=2)

if __name__ == '__main__':
    unittest.main()
```

## 9. Glossary

- **Alpha**: A trading strategy that aims to outperform the market.
- **API (Application Programming Interface)**: A set of protocols for building and integrating application software.
- **Backtest**: A simulation of a trading strategy using historical data.
- **Latency**: The time delay between an action and its effect, crucial in high-frequency trading.
- **Limit Order**: An order to buy or sell at a specific price or better.
- **Liquidity**: The degree to which an asset can be quickly bought or sold without affecting its price.
- **Market Order**: An order to buy or sell immediately at the best available price.
- **Order Book**: A list of buy and sell orders for a specific security or financial instrument.
- **PnL (Profit and Loss)**: The financial benefit or loss from a trade or set of trades.
- **Rebalance**: The process of realigning the weightings of a portfolio of assets.
- **Slippage**: The difference between the expected price of a trade and the price at which the trade is executed.
- **Spread**: The difference between the bid and ask prices of a security.
- **Stop Loss**: An order to sell a security when it reaches a certain price, designed to limit an investor's loss.
- **Volatility**: A statistical measure of the dispersion of returns for a given security or market index.

This comprehensive guide covers the key aspects of using the TWSQ Trading Library, from basic setup to advanced features and best practices. It should serve as a valuable resource for both beginners and experienced quant traders looking to leverage this powerful tool for cryptocurrency trading.