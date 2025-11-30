import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MOMENTUM STRATEGIES
# =============================================================================
class MomentumStrategies:
    """Various momentum-based trading strategies"""
    
    @staticmethod
    def calculate_roc(prices, period):
        """Rate of Change: (Current - Past) / Past"""
        roc = (prices / prices.shift(period) - 1) * 100
        return roc
    
    @staticmethod
    def calculate_momentum(prices, period):
        """Simple momentum: Current - Past"""
        momentum = prices - prices.shift(period)
        return momentum
    
    @staticmethod
    def price_momentum_strategy(df, lookback_days=20, entry_threshold=5):
        """
        BUY: Asset up >X% in past N days
        SELL: Asset down >X% in past N days (or flat)
        Classic "ride the winners" strategy
        """
        df = df.copy()
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        
        # Calculate return over lookback period
        past_price = close.shift(lookback_days)
        momentum_pct = ((close - past_price) / past_price * 100)
        
        # Signal: Buy if up, Sell/Short if down
        df['Momentum_Pct'] = momentum_pct
        df['Signal'] = 0
        df.loc[momentum_pct > entry_threshold, 'Signal'] = 1   # Long
        df.loc[momentum_pct < -entry_threshold, 'Signal'] = -1  # Short
        
        return df
    
    @staticmethod
    def dual_momentum_strategy(df, fast_period=10, slow_period=30):
        """
        Dual Momentum: Compare fast vs slow momentum
        BUY: Fast momentum > Slow momentum (accelerating)
        SELL: Fast momentum < Slow momentum (decelerating)
        """
        df = df.copy()
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        
        # Calculate fast and slow momentum
        fast_mom = MomentumStrategies.calculate_roc(close, fast_period)
        slow_mom = MomentumStrategies.calculate_roc(close, slow_period)
        
        # Signal based on momentum divergence
        df['Fast_Momentum'] = fast_mom
        df['Slow_Momentum'] = slow_mom
        df['Mom_Diff'] = fast_mom - slow_mom
        
        df['Signal'] = 0
        df.loc[df['Mom_Diff'] > 0, 'Signal'] = 1   # Fast > Slow
        df.loc[df['Mom_Diff'] < 0, 'Signal'] = -1  # Fast < Slow
        
        return df
    
    @staticmethod
    def breakout_momentum_strategy(df, lookback=20, num_std=2):
        """
        Breakout with momentum confirmation
        BUY: Price breaks above upper band + rising momentum
        SELL: Price breaks below lower band + falling momentum
        """
        df = df.copy()
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
        low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
        
        # Bollinger Bands
        sma = close.rolling(lookback).mean()
        std = close.rolling(lookback).std()
        upper_band = sma + (num_std * std)
        lower_band = sma - (num_std * std)
        
        # Momentum
        momentum = MomentumStrategies.calculate_roc(close, 10)
        
        # Signal
        df['Signal'] = 0
        df.loc[(close > upper_band) & (momentum > 0), 'Signal'] = 1
        df.loc[(close < lower_band) & (momentum < 0), 'Signal'] = -1
        
        return df
    
    @staticmethod
    def rsm_strategy(df, rsi_period=14, momentum_period=10):
        """
        RSI + Momentum Confirmation
        BUY: RSI < 30 (oversold) AND positive momentum
        SELL: RSI > 70 (overbought) AND negative momentum
        Catches reversals with momentum confirmation
        """
        df = df.copy()
        close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # Momentum
        momentum = MomentumStrategies.calculate_roc(close, momentum_period)
        
        # Signal
        df['RSI'] = rsi
        df['Momentum'] = momentum
        df['Signal'] = 0
        
        # Buy oversold with positive momentum
        df.loc[(rsi < 30) & (momentum > 0), 'Signal'] = 1
        # Sell overbought with negative momentum
        df.loc[(rsi > 70) & (momentum < 0), 'Signal'] = -1
        
        return df


# =============================================================================
# BACKTESTING
# =============================================================================
def calculate_strategy_metrics(df, strategy_col='Strategy_Return_Net'):
    """Calculate performance metrics"""
    returns = df[strategy_col].dropna()
    total_periods = len(returns)
    periods_per_year = 252
    
    metrics = {}
    
    # Returns
    final_value = (1 + returns).prod()
    metrics['Total_Return'] = (final_value - 1) * 100
    metrics['Annualized_Return'] = ((final_value ** (periods_per_year / total_periods)) - 1) * 100 if total_periods > 0 else 0
    
    # Risk
    metrics['Volatility'] = returns.std() * np.sqrt(252) * 100
    metrics['Sharpe'] = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    
    # Drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics['Max_Drawdown'] = drawdown.min() * 100
    
    if metrics['Max_Drawdown'] != 0:
        metrics['Calmar_Ratio'] = metrics['Annualized_Return'] / abs(metrics['Max_Drawdown'])
    else:
        metrics['Calmar_Ratio'] = 0
    
    # Win rate
    metrics['Win_Rate'] = (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0
    
    # Trades
    metrics['Total_Trades'] = df['Position_Change'].sum() if 'Position_Change' in df.columns else 0
    
    return metrics


def backtest_momentum_strategies(ticker='SPY', days_back=1825):
    """Test multiple momentum strategies"""
    
    # Load data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"\n{'='*120}")
    print(f"MOMENTUM STRATEGY BACKTEST: {ticker}")
    print(f"{'='*120}\n")
    
    print(f"Loading {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    print(f"✓ Loaded {len(df)} trading days\n")
    
    # Define strategies to test
    strategies_config = [
        {
            'name': 'Price Momentum (20d, +5%)',
            'func': lambda d: MomentumStrategies.price_momentum_strategy(d, lookback_days=20, entry_threshold=5),
            'description': 'Buy if up >5% in 20 days, short if down >5%'
        },
        {
            'name': 'Price Momentum (10d, +3%)',
            'func': lambda d: MomentumStrategies.price_momentum_strategy(d, lookback_days=10, entry_threshold=3),
            'description': 'Faster: Buy if up >3% in 10 days'
        },
        {
            'name': 'Dual Momentum (10/30)',
            'func': lambda d: MomentumStrategies.dual_momentum_strategy(d, fast_period=10, slow_period=30),
            'description': 'Buy when 10d momentum > 30d momentum'
        },
        {
            'name': 'Breakout + Momentum',
            'func': lambda d: MomentumStrategies.breakout_momentum_strategy(d, lookback=20, num_std=2),
            'description': 'Breakout with momentum confirmation'
        },
        {
            'name': 'RSI + Momentum',
            'func': lambda d: MomentumStrategies.rsm_strategy(d, rsi_period=14, momentum_period=10),
            'description': 'RSI extreme (>70/<30) with momentum confirmation'
        },
    ]
    
    # Run backtests
    results = []
    
    for config in strategies_config:
        print(f"Testing: {config['name']}")
        print(f"  {config['description']}")
        
        df_strat = config['func'](df.copy())
        
        # Calculate returns
        close = df_strat['Close'] if isinstance(df_strat['Close'], pd.Series) else df_strat['Close'].iloc[:, 0]
        df_strat['Returns'] = close.pct_change()
        df_strat['Strategy_Return'] = df_strat['Returns'] * df_strat['Signal'].shift(1)
        
        # Transaction costs
        df_strat['Position_Change'] = df_strat['Signal'].diff().abs()
        transaction_cost = 0.001 + 0.0005
        df_strat['Strategy_Return_Net'] = df_strat['Strategy_Return'] - (df_strat['Position_Change'].shift(1) * transaction_cost)
        
        # Metrics
        metrics = calculate_strategy_metrics(df_strat)
        
        results.append({
            'name': config['name'],
            'df': df_strat,
            'metrics': metrics
        })
        
        print(f"  ✓ Return: {metrics['Total_Return']:>7.2f}% | Sharpe: {metrics['Sharpe']:>6.2f} | "
              f"DD: {metrics['Max_Drawdown']:>7.2f}% | Trades: {metrics['Total_Trades']:>4.0f}\n")
    
    # Benchmark
    benchmark_return = (df['Close'].pct_change() + 1).prod() - 1
    print(f"{'='*120}")
    print(f"BENCHMARK: Buy & Hold = {benchmark_return*100:>7.2f}%")
    print(f"{'='*120}\n")
    
    # Comparison table
    print(f"{'Strategy':<30} {'Return':<12} {'Sharpe':<10} {'Max DD':<12} {'Win %':<10} {'Trades':<10}")
    print("-" * 95)
    
    for result in results:
        metrics = result['metrics']
        print(f"{result['name']:<30} "
              f"{metrics['Total_Return']:>10.2f}% "
              f"{metrics['Sharpe']:>8.2f}  "
              f"{metrics['Max_Drawdown']:>10.2f}% "
              f"{metrics['Win_Rate']:>8.1f}% "
              f"{metrics['Total_Trades']:>8.0f}")
    
    # Best performers
    print(f"\n{'='*120}")
    print("WINNERS")
    print(f"{'='*120}\n")
    
    best_return = max(results, key=lambda x: x['metrics']['Total_Return'])
    best_sharpe = max(results, key=lambda x: x['metrics']['Sharpe'])
    best_drawdown = max(results, key=lambda x: x['metrics']['Calmar_Ratio'])
    
    print(f"🏆 Best Return:      {best_return['name']:<30} {best_return['metrics']['Total_Return']:>7.2f}%")
    print(f"🏆 Best Risk-Adj:    {best_sharpe['name']:<30} Sharpe: {best_sharpe['metrics']['Sharpe']:>6.2f}")
    print(f"🏆 Best Risk Eff:    {best_drawdown['name']:<30} Calmar: {best_drawdown['metrics']['Calmar_Ratio']:>6.2f}")
    
    # Compare to SMA baseline
    print(f"\n{'='*120}")
    print("COMPARISON TO SMA BASELINE (from earlier backtest)")
    print(f"{'='*120}\n")
    
    print(f"SMA (10/30) baseline:              48.23% return | Sharpe: 0.76")
    print(f"Best Momentum strategy:            {best_return['metrics']['Total_Return']:>7.2f}% return | Sharpe: {best_sharpe['metrics']['Sharpe']:>6.2f}")
    print(f"Buy & Hold:                        {benchmark_return*100:>7.2f}% return\n")
    
    # Plot
    print(f"Generating visualization...\n")
    plot_momentum_strategies(results, df, ticker)
    
    print(f"{'='*120}\n")
    
    return results, df


def plot_momentum_strategies(results, df, ticker):
    """Plot equity curves for momentum strategies"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))
    
    # Plot 1: All strategies
    ax = axes[0]
    
    # Benchmark
    benchmark_cum = (1 + df['Close'].pct_change()).cumprod()
    ax.plot(df.index, benchmark_cum, label='Buy & Hold', linewidth=2.5, color='black', linestyle='--', alpha=0.7)
    
    # Strategies
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62246']
    for idx, result in enumerate(results):
        strat_df = result['df']
        cumulative = (1 + strat_df['Strategy_Return_Net']).cumprod()
        ax.plot(strat_df.index, cumulative, label=result['name'], 
               linewidth=2, color=colors[idx % len(colors)], alpha=0.8)
    
    ax.set_title(f'{ticker}: Momentum Strategies vs Buy & Hold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Momentum indicator
    ax = axes[1]
    
    best_strategy = max(results, key=lambda x: x['metrics']['Total_Return'])
    strat_df = best_strategy['df']
    
    close = strat_df['Close'] if isinstance(strat_df['Close'], pd.Series) else strat_df['Close'].iloc[:, 0]
    momentum_20 = MomentumStrategies.calculate_roc(close, 20)
    
    ax.plot(strat_df.index, momentum_20, label='20-Day Momentum (%)', linewidth=2, color='blue')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(5, color='green', linestyle='--', alpha=0.5, label='Buy Threshold (+5%)')
    ax.axhline(-5, color='red', linestyle='--', alpha=0.5, label='Sell Threshold (-5%)')
    ax.fill_between(strat_df.index, momentum_20, 0, where=(momentum_20>0), alpha=0.2, color='green')
    ax.fill_between(strat_df.index, momentum_20, 0, where=(momentum_20<0), alpha=0.2, color='red')
    
    ax.set_title(f'{best_strategy["name"]}: Momentum Indicator', fontsize=11, fontweight='bold')
    ax.set_ylabel('Momentum (%)', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    for ticker in ['SPY', 'QQQ', 'GLD']:
        results, df = backtest_momentum_strategies(ticker, days_back=1825)
