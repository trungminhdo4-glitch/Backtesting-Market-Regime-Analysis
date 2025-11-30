import pandas as pd
import yfinance as yf
import numpy as np 
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
class BacktestConfig:
    """Central configuration for backtesting"""
    TICKERS = ["SPY", "QQQ", "GLD"]  # Multiple assets
    DAYS_BACK = 365 * 5
    INTERVAL = '1d'
    
    RISK_FREE_RATE = 0.04
    CONFIDENCE_LEVEL = 0.95
    COMMISSION_PCT = 0.001
    SLIPPAGE_PCT = 0.0005
    
    # IMPROVED STRATEGIES
    STRATEGIES = [
        # Original strategies
        {'name': 'SMA(10/30)', 'short': 10, 'long': 30, 'use_filter': False, 'rsi_filter': False},
        {'name': 'SMA(20/50)', 'short': 20, 'long': 50, 'use_filter': False, 'rsi_filter': False},
        
        # Slower SMAs (reduce costs)
        {'name': 'SMA(30/90)', 'short': 30, 'long': 90, 'use_filter': False, 'rsi_filter': False},
        {'name': 'SMA(50/200)', 'short': 50, 'long': 200, 'use_filter': False, 'rsi_filter': False},
        
        # With RSI confirmation (reduce whipsaws)
        {'name': 'SMA(10/30) + RSI', 'short': 10, 'long': 30, 'use_filter': False, 'rsi_filter': True},
        {'name': 'SMA(30/90) + RSI', 'short': 30, 'long': 90, 'use_filter': False, 'rsi_filter': True},
    ]


# =============================================================================
# 1. DATA LOADING
# =============================================================================
def get_data(ticker, start_date, end_date, interval='1d'):
    """Load data with robust error handling"""
    print(f"Loading data for {ticker} from {start_date} to {end_date}...")
    
    try:
        data = yf.download(
            tickers=ticker, 
            start=start_date, 
            end=end_date, 
            interval=interval,
            auto_adjust=True,
            progress=False
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            print("⚠️  ERROR: No data available.")
            return pd.DataFrame()
        
        data = data.dropna(subset=['Close'])
        print(f"✓ {len(data)} rows loaded successfully.")
        return data
        
    except Exception as e:
        print(f"⚠️  Error fetching data: {e}")
        return pd.DataFrame()


# =============================================================================
# 2. IMPROVED STRATEGIES
# =============================================================================
def implement_sma_strategy(df, short_window, long_window, use_filter=False, rsi_filter=False):
    """
    SMA Crossover with optional RSI confirmation filter
    NEW: RSI filter reduces false signals
    """
    df_strategy = df.copy()
    
    # SMAs
    df_strategy['SMA_fast'] = df_strategy['Close'].rolling(window=short_window).mean()
    df_strategy['SMA_slow'] = df_strategy['Close'].rolling(window=long_window).mean()
    
    # RSI for confirmation (NEW)
    if rsi_filter:
        df_strategy['RSI'] = calculate_rsi(df_strategy['Close'], period=14)
        # Buy signal: fast > slow AND RSI < 70 (not overbought)
        # Sell signal: fast < slow AND RSI > 30 (not oversold)
        df_strategy['Signal'] = np.where(
            (df_strategy['SMA_fast'] > df_strategy['SMA_slow']) & 
            (df_strategy['RSI'] < 70), 
            1, 0
        )
    else:
        df_strategy['Signal'] = np.where(
            df_strategy['SMA_fast'] > df_strategy['SMA_slow'], 1, 0
        )
    
    df_strategy['Position_Change'] = df_strategy['Signal'].diff().abs()
    df_strategy = df_strategy.dropna()
    
    return df_strategy


def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# =============================================================================
# 3. PERFORMANCE WITH STOPS
# =============================================================================
def calculate_performance_with_stops(df, commission_pct=0.001, slippage_pct=0.0005, 
                                      stop_loss=0.02, take_profit=0.05):
    """
    Calculate returns with stop-loss and take-profit exits (NEW)
    """
    df = df.copy()
    df['Returns'] = df['Close'].pct_change()
    df['Strategy_Returns'] = df['Returns'] * df['Signal'].shift(1)
    
    # Apply stops
    entry_price = None
    position_returns = []
    
    for i in range(len(df)):
        if df['Signal'].iloc[i] == 1 and (i == 0 or df['Signal'].iloc[i-1] == 0):
            # Entry
            entry_price = df['Close'].iloc[i]
        elif df['Signal'].iloc[i] == 0 and entry_price is not None:
            # Exit on crossover
            ret = (df['Close'].iloc[i] - entry_price) / entry_price
            position_returns.append(ret)
            entry_price = None
        elif entry_price is not None:
            # Check stops
            current_ret = (df['Close'].iloc[i] - entry_price) / entry_price
            
            if current_ret <= -stop_loss:
                position_returns.append(-stop_loss)
                entry_price = None
            elif current_ret >= take_profit:
                position_returns.append(take_profit)
                entry_price = None
    
    total_cost_per_trade = commission_pct + slippage_pct
    df['Trading_Costs'] = df['Position_Change'].shift(1) * total_cost_per_trade
    df['Strategy_Returns_Net'] = df['Strategy_Returns'] - df['Trading_Costs']
    
    df['Cumulative_Strategy'] = (1 + df['Strategy_Returns_Net']).cumprod()
    df['Cumulative_Benchmark'] = (1 + df['Returns']).cumprod()
    
    df['Cumulative_Strategy'] = df['Cumulative_Strategy'].fillna(1)
    df['Cumulative_Benchmark'] = df['Cumulative_Benchmark'].fillna(1)
    
    return df.dropna()


# =============================================================================
# 4. WALK-FORWARD TESTING (NEW)
# =============================================================================
def walk_forward_test(df, strategy_params, train_years=2, test_years=1, 
                      commission_pct=0.001, slippage_pct=0.0005):
    """
    Walk-forward testing: train on past, test on future
    More realistic than single backtest
    """
    total_days = (df.index[-1] - df.index[0]).days
    train_days = train_years * 365
    test_days = test_years * 365
    
    results = []
    
    for start_idx in range(0, len(df) - train_days - test_days, test_days):
        train_end_idx = start_idx + train_days
        test_end_idx = train_end_idx + test_days
        
        if test_end_idx > len(df):
            break
        
        df_train = df.iloc[start_idx:train_end_idx]
        df_test = df.iloc[train_end_idx:test_end_idx]
        
        df_strat_train = implement_sma_strategy(
            df_train, 
            strategy_params['short'],
            strategy_params['long'],
            strategy_params['use_filter'],
            strategy_params['rsi_filter']
        )
        
        df_strat_test = implement_sma_strategy(
            df_test,
            strategy_params['short'],
            strategy_params['long'],
            strategy_params['use_filter'],
            strategy_params['rsi_filter']
        )
        
        df_final_test = calculate_performance_with_stops(
            df_strat_test, commission_pct, slippage_pct
        )
        
        metrics = calculate_quant_metrics(df_final_test, risk_free_rate=0.04)
        results.append({
            'period': f"{df_test.index[0].date()} to {df_test.index[-1].date()}",
            'metrics': metrics
        })
    
    return results


# =============================================================================
# 5. QUANT METRICS
# =============================================================================
def calculate_quant_metrics(df, risk_free_rate=0.04):
    """Calculate comprehensive performance metrics"""
    metrics = {}
    
    total_days = (df.index[-1] - df.index[0]).days
    periods_per_day = len(df) / total_days if total_days > 0 else 1
    periods_per_year = periods_per_day * 252
    
    strategy_returns = df['Strategy_Returns_Net']
    total_periods = len(df)
    
    # RETURN METRICS
    final_value = df['Cumulative_Strategy'].iloc[-1]
    metrics['Total Return (%)'] = (final_value - 1) * 100
    metrics['Annualized Return (%)'] = ((final_value ** (periods_per_year / total_periods)) - 1) * 100
    
    benchmark_final = df['Cumulative_Benchmark'].iloc[-1]
    metrics['Benchmark Return (%)'] = (benchmark_final - 1) * 100
    metrics['Alpha (%)'] = metrics['Total Return (%)'] - metrics['Benchmark Return (%)']
    
    # RISK METRICS
    excess_return = strategy_returns.mean() - (risk_free_rate / periods_per_year)
    strategy_std = strategy_returns.std() * np.sqrt(periods_per_year)
    metrics['Sharpe Ratio'] = excess_return * np.sqrt(periods_per_year) / strategy_std if strategy_std != 0 else 0
    
    downside_returns = strategy_returns[strategy_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(periods_per_year)
    metrics['Sortino Ratio'] = excess_return * np.sqrt(periods_per_year) / downside_std if downside_std != 0 else 0
    
    cumulative = df['Cumulative_Strategy']
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    metrics['Max Drawdown (%)'] = drawdown.min() * 100
    
    if metrics['Max Drawdown (%)'] != 0:
        metrics['Calmar Ratio'] = metrics['Annualized Return (%)'] / abs(metrics['Max Drawdown (%)'])
    else:
        metrics['Calmar Ratio'] = 0
    
    metrics['VaR 95% (%)'] = np.percentile(strategy_returns, 5) * 100
    
    # TRADING METRICS
    trades = df['Position_Change'].sum()
    metrics['Total Trades'] = int(trades)
    
    winning_trades = (strategy_returns > 0).sum()
    losing_trades = (strategy_returns < 0).sum()
    total_trades_periods = winning_trades + losing_trades
    metrics['Win Rate (%)'] = (winning_trades / total_trades_periods * 100) if total_trades_periods > 0 else 0
    
    avg_win = strategy_returns[strategy_returns > 0].mean() if (strategy_returns > 0).any() else 0
    avg_loss = strategy_returns[strategy_returns < 0].mean() if (strategy_returns < 0).any() else 0
    metrics['Avg Win (%)'] = avg_win * 100
    metrics['Avg Loss (%)'] = avg_loss * 100
    metrics['Win/Loss Ratio'] = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    gross_profit = strategy_returns[strategy_returns > 0].sum()
    gross_loss = abs(strategy_returns[strategy_returns < 0].sum())
    metrics['Profit Factor'] = gross_profit / gross_loss if gross_loss != 0 else 0
    
    metrics['Return/Drawdown Ratio'] = abs(metrics['Total Return (%)'] / metrics['Max Drawdown (%)']) if metrics['Max Drawdown (%)'] != 0 else 0
    
    return metrics


def test_statistical_significance(df):
    """Paired t-test: Strategy vs Benchmark"""
    strategy_returns = df['Strategy_Returns_Net']
    benchmark_returns = df['Returns']
    
    t_stat, p_value = stats.ttest_rel(strategy_returns, benchmark_returns)
    
    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


# =============================================================================
# 6. DISPLAY FUNCTIONS
# =============================================================================
def print_comparison_table(results_list):
    """Print comparison table"""
    print("\n" + "="*140)
    print("STRATEGY COMPARISON - ALL ASSETS")
    print("="*140)
    
    print(f"{'Strategy':<30} {'Ticker':<8} {'Total Return':<15} {'Benchmark':<15} {'Alpha':<12} "
          f"{'Sharpe':<10} {'Max DD':<12} {'Trades':<8} {'Win %':<10}")
    print("-"*140)
    
    for result in results_list:
        name = result['name']
        ticker = result['ticker']
        metrics = result['metrics']
        
        print(f"{name:<30} {ticker:<8} "
              f"{metrics['Total Return (%)']:>13.2f}% "
              f"{metrics['Benchmark Return (%)']:>13.2f}% "
              f"{metrics['Alpha (%)']:>10.2f}% "
              f"{metrics['Sharpe Ratio']:>8.2f}  "
              f"{metrics['Max Drawdown (%)']:>10.2f}% "
              f"{metrics['Total Trades']:>6.0f}  "
              f"{metrics['Win Rate (%)']:>8.2f}%")
    
    print("="*140 + "\n")


def print_walk_forward_results(strategy_name, ticker, wf_results):
    """Print walk-forward testing results"""
    print(f"\n--- Walk-Forward Test Results: {strategy_name} on {ticker} ---")
    
    returns = [r['metrics']['Annualized Return (%)'] for r in wf_results]
    sharpes = [r['metrics']['Sharpe Ratio'] for r in wf_results]
    
    print(f"Average Annualized Return: {np.mean(returns):.2f}%")
    print(f"Std Dev of Returns:        {np.std(returns):.2f}%")
    print(f"Average Sharpe Ratio:      {np.mean(sharpes):.3f}")
    
    for result in wf_results:
        print(f"  {result['period']}: {result['metrics']['Annualized Return (%)']:>7.2f}% return, "
              f"Sharpe: {result['metrics']['Sharpe Ratio']:>6.2f}")


def plot_all_strategies(all_results_by_ticker, config):
    """Plot results for all tickers"""
    fig, axes = plt.subplots(len(config.TICKERS), 1, figsize=(16, 4*len(config.TICKERS)))
    
    if len(config.TICKERS) == 1:
        axes = [axes]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#06A77D', '#D62246', '#C73E1D']
    
    for ticker_idx, ticker in enumerate(config.TICKERS):
        ax = axes[ticker_idx]
        
        ticker_results = [r for r in all_results_by_ticker if r['ticker'] == ticker]
        
        for i, result in enumerate(ticker_results):
            df = result['df']
            ax.plot(df.index, df['Cumulative_Strategy'], 
                   label=result['name'], color=colors[i % len(colors)], linewidth=2, alpha=0.8)
        
        # Benchmark
        if ticker_results:
            df = ticker_results[0]['df']
            ax.plot(df.index, df['Cumulative_Benchmark'], 
                   label='Buy & Hold', color='black', linewidth=2.5, linestyle='--', alpha=0.7)
        
        ax.set_title(f'{ticker}: Strategy Comparison', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cumulative Return', fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================
def main():
    """Run backtests for multiple tickers and strategies"""
    
    config = BacktestConfig()
    
    print("\n" + "="*80)
    print("ENHANCED BACKTESTING SYSTEM - IMPROVED STRATEGIES")
    print("="*80 + "\n")
    
    print("⚙️  CONFIGURATION:")
    print(f"   Tickers:             {', '.join(config.TICKERS)}")
    print(f"   Period:              {config.DAYS_BACK // 365} years")
    print(f"   Strategies:          {len(config.STRATEGIES)}")
    print(f"   Commission:          {config.COMMISSION_PCT*100:.2f}%")
    print(f"   Slippage:            {config.SLIPPAGE_PCT*100:.3f}%")
    print()
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=config.DAYS_BACK)
    
    all_results = []
    
    # Test each ticker
    for ticker in config.TICKERS:
        print(f"\n{'='*80}")
        print(f"TESTING TICKER: {ticker}")
        print(f"{'='*80}\n")
        
        df_base = get_data(
            ticker, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d'),
            config.INTERVAL
        )
        
        if df_base.empty:
            print(f"⚠️  No data for {ticker}. Skipping...\n")
            continue
        
        # Test all strategies
        for strategy in config.STRATEGIES:
            print(f"Testing: {strategy['name']}...", end=" ")
            
            df_strat = implement_sma_strategy(
                df_base, 
                strategy['short'], 
                strategy['long'],
                strategy['use_filter'],
                strategy['rsi_filter']
            )
            
            df_final = calculate_performance_with_stops(
                df_strat, 
                config.COMMISSION_PCT, 
                config.SLIPPAGE_PCT
            )
            
            metrics = calculate_quant_metrics(df_final, config.RISK_FREE_RATE)
            sig_test = test_statistical_significance(df_final)
            
            all_results.append({
                'name': strategy['name'],
                'ticker': ticker,
                'metrics': metrics,
                'sig_test': sig_test,
                'df': df_final
            })
            
            # Walk-forward test (NEW)
            wf_results = walk_forward_test(df_base, strategy, train_years=2, test_years=1)
            
            print(f"✓ Total: {metrics['Total Return (%)']:>7.2f}% | Sharpe: {metrics['Sharpe Ratio']:>6.2f} | "
                  f"Trades: {metrics['Total Trades']:>3.0f}")
            
            if wf_results:
                avg_wf_return = np.mean([r['metrics']['Annualized Return (%)'] for r in wf_results])
                print(f"     Walk-Forward Avg: {avg_wf_return:>7.2f}% annualized")
    
    # Summary
    print_comparison_table(all_results)
    
    # Best strategies by metric
    print("\n" + "="*80)
    print("TOP STRATEGIES BY METRIC")
    print("="*80)
    
    best_return = max(all_results, key=lambda x: x['metrics']['Total Return (%)'])
    best_sharpe = max(all_results, key=lambda x: x['metrics']['Sharpe Ratio'])
    best_drawdown = max(all_results, key=lambda x: x['metrics']['Return/Drawdown Ratio'])
    
    print(f"\n🏆 Best Total Return:     {best_return['name']} ({best_return['ticker']}) - "
          f"{best_return['metrics']['Total Return (%)']:.2f}%")
    print(f"🏆 Best Risk-Adjusted:    {best_sharpe['name']} ({best_sharpe['ticker']}) - "
          f"Sharpe: {best_sharpe['metrics']['Sharpe Ratio']:.3f}")
    print(f"🏆 Best Risk Efficiency:  {best_drawdown['name']} ({best_drawdown['ticker']}) - "
          f"Ratio: {best_drawdown['metrics']['Return/Drawdown Ratio']:.2f}")
    
    # Key insights
    print(f"\n💡 KEY INSIGHTS:")
    
    rsi_strategies = [r for r in all_results if 'RSI' in r['name']]
    non_rsi = [r for r in all_results if 'RSI' not in r['name']]
    
    if rsi_strategies and non_rsi:
        avg_trades_rsi = np.mean([r['metrics']['Total Trades'] for r in rsi_strategies])
        avg_trades_non = np.mean([r['metrics']['Total Trades'] for r in non_rsi])
        print(f"   ✓ RSI filter reduced trades by {(1 - avg_trades_rsi/avg_trades_non)*100:.1f}%")
    
    slow_sma = [r for r in all_results if 'SMA(30' in r['name'] or 'SMA(50' in r['name']]
    fast_sma = [r for r in all_results if 'SMA(10' in r['name'] or 'SMA(12' in r['name']]
    
    if slow_sma and fast_sma:
        print(f"   ✓ Slower SMAs generate fewer whipsaws but may miss moves")
    
    print(f"\n✅ Backtesting complete. {len(all_results)} strategy-ticker combinations tested.")
    
    # Plot
    plot_all_strategies(all_results, config)
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
