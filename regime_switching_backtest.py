import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# REGIME DETECTION
# =============================================================================
class RegimeDetector:
    """Detect market regimes in real-time"""
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Calculate ADX"""
        if isinstance(high, pd.DataFrame):
            high = high.iloc[:, 0]
        if isinstance(low, pd.DataFrame):
            low = low.iloc[:, 0]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
        
        tr = np.maximum(high - low, np.maximum(
            abs(high - close.shift(1)), abs(low - close.shift(1))
        ))
        tr_sum = tr.rolling(period).sum()
        
        plus_di = 100 * (plus_dm.rolling(period).sum() / tr_sum)
        minus_di = 100 * (minus_dm.rolling(period).sum() / tr_sum)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def get_trend_regime(adx, strong_threshold=25, weak_threshold=20):
        """
        NO_TREND: ADX < 20
        WEAK_TREND: 20 <= ADX < 25
        STRONG_TREND: ADX >= 25
        """
        regime = pd.Series('NO_TREND', index=adx.index)
        regime[adx >= strong_threshold] = 'STRONG_TREND'
        regime[(adx >= weak_threshold) & (adx < strong_threshold)] = 'WEAK_TREND'
        return regime
    
    @staticmethod
    def get_volatility_regime(returns, period=20):
        """LOW_VOL, MED_VOL, HIGH_VOL based on percentiles"""
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        
        volatility = returns.rolling(period).std()
        q33 = volatility.rolling(252).quantile(0.33)
        q67 = volatility.rolling(252).quantile(0.67)
        
        regime = pd.Series('MED_VOL', index=returns.index)
        regime[volatility < q33] = 'LOW_VOL'
        regime[volatility > q67] = 'HIGH_VOL'
        return regime


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================
def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def sma_trend_signal(df, short=10, long=30):
    """SMA crossover signal for trending markets"""
    close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    sma_fast = close.rolling(short).mean()
    sma_slow = close.rolling(long).mean()
    
    signal = pd.Series(0, index=df.index)
    signal[sma_fast > sma_slow] = 1
    
    return signal, sma_fast, sma_slow


def mean_reversion_signal(df, period=20, z_threshold=2.0):
    """Mean reversion signal for ranging markets"""
    close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    z_score = (close - sma) / std
    
    signal = pd.Series(0, index=df.index)
    signal[z_score < -z_threshold] = 1   # Oversold - buy
    signal[z_score > z_threshold] = -1   # Overbought - short (we'll just exit)
    
    return signal, z_score


def regime_switching_strategy(df):
    """
    Smart Vol-Based Exit: Trade SMA in LOW/MED vol, EXIT entirely in HIGH vol
    LOW_VOL: 1.0x (full)
    MED_VOL: 0.5x (half)
    HIGH_VOL: 0.0x (exit - hold cash)
    """
    df = df.copy()
    close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    # Calculate regimes
    adx, plus_di, minus_di = RegimeDetector.calculate_adx(
        df['High'], df['Low'], df['Close']
    )
    trend_regime = RegimeDetector.get_trend_regime(adx)
    vol_regime = RegimeDetector.get_volatility_regime(close.pct_change())
    
    # Generate SMA signal (base signal)
    sma_sig, sma_fast, sma_slow = sma_trend_signal(df, short=10, long=30)
    
    # Position sizing based on volatility - SMART EXIT in high vol
    # LOW_VOL: full position (1.0x), MED_VOL: 0.5x, HIGH_VOL: 0.0x (exit)
    position_size = pd.Series(0.5, index=df.index)  # Default medium
    position_size[vol_regime == 'LOW_VOL'] = 1.0    # Full in calm
    position_size[vol_regime == 'HIGH_VOL'] = 0.0   # Exit in chaos
    
    # Apply position sizing to signal
    df['Signal'] = sma_sig * position_size
    df['Active_Strategy'] = 'SMA_SmartExit'
    
    # Track which regime we're in
    df['Trend_Regime'] = trend_regime
    df['Volatility_Regime'] = vol_regime
    
    # Calculate returns
    df['Returns'] = close.pct_change()
    df['Strategy_Return'] = df['Returns'] * df['Signal'].shift(1)
    
    # Add transaction costs
    df['Position_Change'] = df['Signal'].diff().abs()
    transaction_cost = 0.001 + 0.0005  # commission + slippage
    df['Strategy_Return_Net'] = df['Strategy_Return'] - (df['Position_Change'].shift(1) * transaction_cost)
    
    # Store regime data
    df['ADX'] = adx
    df['SMA_Fast'] = sma_fast
    df['SMA_Slow'] = sma_slow
    
    return df


# =============================================================================
# COMPARISON: Pure SMA vs. Regime-Switching
# =============================================================================
def pure_sma_strategy(df):
    """Baseline: SMA only (your original strategy)"""
    df = df.copy()
    close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    sma_fast = close.rolling(10).mean()
    sma_slow = close.rolling(30).mean()
    
    df['Signal'] = np.where(sma_fast > sma_slow, 1, 0)
    df['Returns'] = close.pct_change()
    df['Strategy_Return'] = df['Returns'] * df['Signal'].shift(1)
    
    # Transaction costs
    df['Position_Change'] = df['Signal'].diff().abs()
    transaction_cost = 0.001 + 0.0005
    df['Strategy_Return_Net'] = df['Strategy_Return'] - (df['Position_Change'].shift(1) * transaction_cost)
    
    return df


# =============================================================================
# METRICS
# =============================================================================
def calculate_metrics(df, strategy_col='Strategy_Return_Net'):
    """Calculate comprehensive metrics"""
    returns = df[strategy_col].dropna()
    total_days = (df.index[-1] - df.index[0]).days
    total_periods = len(returns)
    periods_per_year = 252
    
    metrics = {}
    
    # Returns
    final_value = (1 + returns).prod()
    metrics['Total_Return'] = (final_value - 1) * 100
    metrics['Annualized_Return'] = ((final_value ** (periods_per_year / total_periods)) - 1) * 100
    
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
    metrics['Win_Rate'] = (returns > 0).sum() / len(returns) * 100
    
    # Trades
    metrics['Total_Trades'] = df['Position_Change'].sum()
    
    return metrics


def compare_strategies(ticker, days_back=1825):
    """Compare Pure SMA vs. Regime-Switching"""
    
    # Load data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"\n{'='*100}")
    print(f"REGIME-SWITCHING BACKTEST: {ticker}")
    print(f"{'='*100}\n")
    
    print(f"Loading {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    print(f"✓ Loaded {len(df)} trading days\n")
    
    # Run strategies
    print("Running Pure SMA strategy...")
    df_sma = pure_sma_strategy(df.copy())
    sma_metrics = calculate_metrics(df_sma)
    print(f"✓ SMA Total Return: {sma_metrics['Total_Return']:>7.2f}% | Sharpe: {sma_metrics['Sharpe']:>6.2f}")
    
    print("Running Regime-Switching strategy...")
    df_regime = regime_switching_strategy(df.copy())
    regime_metrics = calculate_metrics(df_regime)
    print(f"✓ Regime Total Return: {regime_metrics['Total_Return']:>7.2f}% | Sharpe: {regime_metrics['Sharpe']:>6.2f}")
    
    # Benchmark
    benchmark_return = (df_sma['Returns'].dropna() + 1).prod() - 1
    print(f"✓ Buy & Hold Return: {benchmark_return*100:>7.2f}%\n")
    
    # Detailed comparison
    print(f"{'='*100}")
    print(f"DETAILED COMPARISON")
    print(f"{'='*100}\n")
    
    print(f"{'Metric':<25} {'Pure SMA':<20} {'Regime-Switching':<20} {'Difference':<15}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('Total Return (%)', 'Total_Return'),
        ('Annualized Return (%)', 'Annualized_Return'),
        ('Volatility (%)', 'Volatility'),
        ('Sharpe Ratio', 'Sharpe'),
        ('Max Drawdown (%)', 'Max_Drawdown'),
        ('Calmar Ratio', 'Calmar_Ratio'),
        ('Win Rate (%)', 'Win_Rate'),
        ('Total Trades', 'Total_Trades'),
    ]
    
    for display_name, metric_key in metrics_to_compare:
        sma_val = sma_metrics[metric_key]
        regime_val = regime_metrics[metric_key]
        diff = regime_val - sma_val
        
        print(f"{display_name:<25} {sma_val:>18.2f} {regime_val:>18.2f} {diff:>+13.2f}")
    
    print(f"\n{'='*100}")
    print("REGIME STATISTICS")
    print(f"{'='*100}\n")
    
    # Volatility regime breakdown
    low_vol = (df_regime['Volatility_Regime'] == 'LOW_VOL').sum()
    med_vol = (df_regime['Volatility_Regime'] == 'MED_VOL').sum()
    high_vol = (df_regime['Volatility_Regime'] == 'HIGH_VOL').sum()
    total_days = low_vol + med_vol + high_vol
    
    print(f"Days in LOW_VOL:        {low_vol:>6.0f} ({low_vol/total_days*100:>5.1f}%) - Full 1.0x position size")
    print(f"Days in MED_VOL:        {med_vol:>6.0f} ({med_vol/total_days*100:>5.1f}%) - 0.5x position size")
    print(f"Days in HIGH_VOL:       {high_vol:>6.0f} ({high_vol/total_days*100:>5.1f}%) - 0.0x position size (EXIT)")
    
    print(f"\nPerformance by volatility regime:")
    
    for vol_type, position_mult in [('LOW_VOL', '1.0x'), ('MED_VOL', '0.5x'), ('HIGH_VOL', '0.0x')]:
        vol_periods = df_regime[df_regime['Volatility_Regime'] == vol_type]['Strategy_Return_Net']
        if len(vol_periods) > 0:
            vol_return = ((1 + vol_periods).prod() - 1) * 100
            vol_sharpe = vol_periods.mean() / vol_periods.std() * np.sqrt(252) if vol_periods.std() > 0 else 0
            print(f"  {vol_type} ({position_mult}):   {vol_return:>7.2f}% return | Sharpe: {vol_sharpe:>6.2f}")
    
    # Plot
    print(f"\nGenerating visualization...\n")
    plot_comparison(df_sma, df_regime, ticker)
    
    print(f"{'='*100}\n")
    
    return df_sma, df_regime, sma_metrics, regime_metrics


def plot_comparison(df_sma, df_regime, ticker):
    """Plot equity curves and regimes"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # Prepare cumulative returns
    df_sma['Cumulative_SMA'] = (1 + df_sma['Strategy_Return_Net']).cumprod()
    df_sma['Cumulative_Benchmark'] = (1 + df_sma['Returns']).cumprod()
    
    df_regime['Cumulative_Regime'] = (1 + df_regime['Strategy_Return_Net']).cumprod()
    
    # Plot 1: Equity curves
    ax = axes[0]
    ax.plot(df_sma.index, df_sma['Cumulative_SMA'], label='Pure SMA', linewidth=2.5, color='blue', alpha=0.8)
    ax.plot(df_regime.index, df_regime['Cumulative_Regime'], label='Regime-Switching', linewidth=2.5, color='green', alpha=0.8)
    ax.plot(df_sma.index, df_sma['Cumulative_Benchmark'], label='Buy & Hold', linewidth=2.5, 
            color='black', linestyle='--', alpha=0.6)
    
    ax.set_title(f'{ticker}: Strategy Comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Return', fontsize=11)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: ADX and active strategy
    ax = axes[1]
    ax.plot(df_regime.index, df_regime['ADX'], linewidth=2, color='black', label='ADX (Trend Strength)')
    ax.axhline(25, color='red', linestyle='--', alpha=0.5, label='ADX=25 (Threshold)')
    ax.axhline(20, color='orange', linestyle='--', alpha=0.5)
    
    # Color background by strategy
    sma_periods = df_regime[df_regime['Active_Strategy'] == 'SMA'].index
    mr_periods = df_regime[df_regime['Active_Strategy'] == 'MeanReversion'].index
    
    for idx in sma_periods:
        ax.axvspan(idx, idx + timedelta(days=1), alpha=0.05, color='blue')
    for idx in mr_periods:
        ax.axvspan(idx, idx + timedelta(days=1), alpha=0.05, color='red')
    
    ax.set_title('ADX & Active Strategy (Blue=SMA, Red=MeanReversion)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ADX Value', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Rolling Sharpe ratio
    ax = axes[2]
    window = 60
    sma_sharpe = (df_sma['Strategy_Return_Net'].rolling(window).mean() / 
                  df_sma['Strategy_Return_Net'].rolling(window).std() * np.sqrt(252))
    regime_sharpe = (df_regime['Strategy_Return_Net'].rolling(window).mean() / 
                     df_regime['Strategy_Return_Net'].rolling(window).std() * np.sqrt(252))
    
    ax.plot(df_sma.index, sma_sharpe, label='Pure SMA', linewidth=2, color='blue', alpha=0.8)
    ax.plot(df_regime.index, regime_sharpe, label='Regime-Switching', linewidth=2, color='green', alpha=0.8)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    
    ax.set_title(f'Rolling {window}-Day Sharpe Ratio', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sharpe Ratio', fontsize=10)
    ax.set_xlabel('Date', fontsize=10)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Test on multiple assets
    for ticker in ['SPY', 'QQQ', 'GLD']:
        df_sma, df_regime, sma_metrics, regime_metrics = compare_strategies(ticker, days_back=1825)
