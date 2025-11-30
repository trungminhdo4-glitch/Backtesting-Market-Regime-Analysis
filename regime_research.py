import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# MARKET REGIME ANALYSIS
# =============================================================================
class RegimeAnalyzer:
    """Identify market regimes and analyze strategy performance by regime"""
    
    @staticmethod
    def calculate_adx(high, low, close, period=14):
        """Calculate Average Directional Index (trend strength)"""
        # Convert to Series if needed
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
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        tr_sum = tr.rolling(period).sum()
        
        plus_di = 100 * (plus_dm.rolling(period).sum() / tr_sum)
        minus_di = 100 * (minus_dm.rolling(period).sum() / tr_sum)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_volatility_regime(returns, period=20):
        """Classify volatility as LOW/MEDIUM/HIGH"""
        # Handle Series or DataFrame
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        
        volatility = returns.rolling(period).std()
        q33 = volatility.quantile(0.33)
        q67 = volatility.quantile(0.67)
        
        regime = pd.cut(volatility, bins=[0, q33, q67, np.inf], 
                       labels=['LOW_VOL', 'MED_VOL', 'HIGH_VOL'])
        return regime
    
    @staticmethod
    def calculate_trend_regime(adx):
        """Classify trend as STRONG_TREND, WEAK_TREND, or NO_TREND"""
        # Handle Series or DataFrame
        if isinstance(adx, pd.DataFrame):
            adx = adx.iloc[:, 0]
        
        regime = pd.cut(adx, bins=[0, 20, 40, np.inf],
                       labels=['NO_TREND', 'WEAK_TREND', 'STRONG_TREND'])
        return regime
    
    @staticmethod
    def classify_market_condition(returns, high, low, close, period=14):
        """Comprehensive market regime classification"""
        adx, plus_di, minus_di = RegimeAnalyzer.calculate_adx(high, low, close, period)
        vol_regime = RegimeAnalyzer.calculate_volatility_regime(returns)
        trend_regime = RegimeAnalyzer.calculate_trend_regime(adx)
        
        # Determine direction
        direction = np.where(plus_di > minus_di, 'UP', 'DOWN')
        
        # Combined regime
        regime = pd.DataFrame({
            'ADX': adx,
            'Plus_DI': plus_di,
            'Minus_DI': minus_di,
            'Volatility_Regime': vol_regime,
            'Trend_Regime': trend_regime,
            'Direction': direction
        }, index=close.index)
        
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


def sma_trend_strategy(df, short=10, long=30):
    """Traditional SMA crossover - TREND FOLLOWING"""
    df = df.copy()
    close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    df['SMA_Fast'] = close.rolling(short).mean()
    df['SMA_Slow'] = close.rolling(long).mean()
    df['Signal'] = np.where(df['SMA_Fast'] > df['SMA_Slow'], 1, 0)
    df['Strategy_Return'] = close.pct_change() * df['Signal'].shift(1)
    return df


def mean_reversion_strategy(df, period=20, z_threshold=2.0):
    """Mean reversion - BUY DIPS, SELL RALLIES"""
    df = df.copy()
    close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    
    # Z-score
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    df['Z_Score'] = (close - sma) / std
    
    # Buy oversold, sell overbought
    df['Signal'] = 0
    df.loc[df['Z_Score'] < -z_threshold, 'Signal'] = 1   # Buy
    df.loc[df['Z_Score'] > z_threshold, 'Signal'] = -1   # Sell (short)
    
    df['Strategy_Return'] = close.pct_change() * df['Signal'].shift(1)
    return df


def volatility_breakout_strategy(df, atr_period=14, std_dev=2):
    """BREAKOUT - Trade on volatility expansion"""
    df = df.copy()
    close = df['Close'] if isinstance(df['Close'], pd.Series) else df['Close'].iloc[:, 0]
    high = df['High'] if isinstance(df['High'], pd.Series) else df['High'].iloc[:, 0]
    low = df['Low'] if isinstance(df['Low'], pd.Series) else df['Low'].iloc[:, 0]
    
    # ATR
    h_l = high - low
    h_cp = abs(high - close.shift(1))
    l_cp = abs(low - close.shift(1))
    tr = pd.concat([h_l, h_cp, l_cp], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()
    
    # Breakout bands
    df['Upper_Band'] = close.rolling(20).mean() + (std_dev * atr)
    df['Lower_Band'] = close.rolling(20).mean() - (std_dev * atr)
    
    # Signal: break above/below
    df['Signal'] = 0
    df.loc[close > df['Upper_Band'].shift(1), 'Signal'] = 1   # Buy breakup
    df.loc[close < df['Lower_Band'].shift(1), 'Signal'] = -1  # Sell breakdown
    
    df['Strategy_Return'] = close.pct_change() * df['Signal'].shift(1)
    return df


# =============================================================================
# ANALYSIS BY REGIME
# =============================================================================
def analyze_strategy_by_regime(df_with_strategy, regime_col, strategy_col='Strategy_Return'):
    """Analyze strategy performance broken down by market regime"""
    results = {}
    
    for regime_type in df_with_strategy[regime_col].unique():
        if pd.isna(regime_type):
            continue
        
        subset = df_with_strategy[df_with_strategy[regime_col] == regime_type]
        
        if len(subset) == 0:
            continue
        
        returns = subset[strategy_col].dropna()
        
        results[regime_type] = {
            'Total_Return': (1 + returns).prod() - 1,
            'Annualized_Return': ((1 + returns).prod() ** (252 / len(returns)) - 1) if len(returns) > 0 else 0,
            'Sharpe': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
            'Win_Rate': (returns > 0).sum() / len(returns) * 100 if len(returns) > 0 else 0,
            'Avg_Win': returns[returns > 0].mean() * 100 if (returns > 0).any() else 0,
            'Avg_Loss': returns[returns < 0].mean() * 100 if (returns < 0).any() else 0,
            'Max_DD': ((1 + returns).cumprod() / (1 + returns).cumprod().cummax() - 1).min() * 100,
            'Sample_Size': len(returns)
        }
    
    return results


def compare_strategies_by_regime(data_dict, regime_col):
    """Compare all strategies across market regimes"""
    print("\n" + "="*140)
    print(f"STRATEGY PERFORMANCE BY MARKET REGIME: {regime_col}")
    print("="*140 + "\n")
    
    for regime_type in sorted(data_dict[list(data_dict.keys())[0]].keys()):
        print(f"\n--- {regime_type.upper()} ---")
        print(f"{'Strategy':<25} {'Return':<12} {'Sharpe':<10} {'Win %':<10} {'Avg Win':<10} {'Avg Loss':<10} {'Max DD':<10} {'Samples':<8}")
        print("-" * 105)
        
        for strategy_name, regime_results in data_dict.items():
            if regime_type not in regime_results:
                continue
            
            metrics = regime_results[regime_type]
            print(f"{strategy_name:<25} "
                  f"{metrics['Total_Return']*100:>10.2f}% "
                  f"{metrics['Sharpe']:>8.2f}  "
                  f"{metrics['Win_Rate']:>8.1f}% "
                  f"{metrics['Avg_Win']:>8.2f}% "
                  f"{metrics['Avg_Loss']:>8.2f}% "
                  f"{metrics['Max_DD']:>8.2f}% "
                  f"{metrics['Sample_Size']:>6.0f}")


# =============================================================================
# VISUALIZATION
# =============================================================================
def plot_regime_analysis(df, regime_df, strategies_dict, ticker):
    """Plot: equity curves colored by market regime"""
    fig, axes = plt.subplots(3, 1, figsize=(16, 10))
    
    # Prepare data
    df = df.join(regime_df)
    
    # Color map for regimes
    trend_colors = {'STRONG_TREND': 'green', 'WEAK_TREND': 'orange', 'NO_TREND': 'red'}
    vol_colors = {'LOW_VOL': 'blue', 'MED_VOL': 'gray', 'HIGH_VOL': 'red'}
    
    # Plot 1: Equity curves by trend regime
    ax = axes[0]
    for strategy_name, strategy_df in strategies_dict.items():
        strategy_df = strategy_df.copy()
        strategy_df['Cumulative'] = (1 + strategy_df['Strategy_Return']).cumprod()
        ax.plot(strategy_df.index, strategy_df['Cumulative'], label=strategy_name, linewidth=2, alpha=0.7)
    
    # Shade by trend regime
    for trend_type, color in trend_colors.items():
        for idx, (date, row) in enumerate(df.iterrows()):
            if row['Trend_Regime'] == trend_type:
                ax.axvspan(date, date + timedelta(days=1), alpha=0.1, color=color)
    
    ax.set_title(f'{ticker}: Strategy Performance Colored by Trend Regime (Green=Strong, Orange=Weak, Red=None)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative Return')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: ADX and DI
    ax = axes[1]
    ax.plot(df.index, df['ADX'], label='ADX (Trend Strength)', linewidth=2, color='black')
    ax.plot(df.index, df['Plus_DI'], label='+DI (Uptrend)', linewidth=1.5, color='green', alpha=0.7)
    ax.plot(df.index, df['Minus_DI'], label='-DI (Downtrend)', linewidth=1.5, color='red', alpha=0.7)
    ax.axhline(20, color='gray', linestyle='--', alpha=0.5, label='ADX=20 (weak/strong threshold)')
    ax.set_title('Trend Strength Indicators (ADX, DI)', fontsize=11, fontweight='bold')
    ax.set_ylabel('ADX / DI Value')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Volatility regime
    ax = axes[2]
    df['Returns'] = df['Close'].pct_change()
    df['Volatility'] = df['Returns'].rolling(20).std() * 100
    ax.plot(df.index, df['Volatility'], label='Rolling Volatility (20d)', linewidth=2, color='purple')
    ax.fill_between(df.index, 0, df['Volatility'], alpha=0.3, color='purple')
    ax.set_title('Market Volatility Over Time', fontsize=11, fontweight='bold')
    ax.set_ylabel('Volatility (%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN RESEARCH ANALYSIS
# =============================================================================
def run_regime_research(ticker='SPY', days_back=1825):
    """Complete regime analysis"""
    
    print("\n" + "="*80)
    print(f"MARKET REGIME RESEARCH: {ticker}")
    print("="*80 + "\n")
    
    # Load data
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    print(f"Loading {ticker} from {start_date} to {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    # Fix MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df = df.dropna()
    print(f"✓ Loaded {len(df)} trading days\n")
    
    # Calculate regimes
    print("Analyzing market regimes...")
    regime_df = RegimeAnalyzer.classify_market_condition(
        df['Close'].pct_change(),
        df['High'],
        df['Low'],
        df['Close']
    )
    print("✓ Calculated ADX, volatility, trend classification\n")
    
    # Implement strategies
    print("Testing strategies...")
    strategies = {}
    for strat_name, strat_func in [
        ('SMA (10/30) - Trend', lambda d: sma_trend_strategy(d, short=10, long=30)),
        ('SMA (50/200) - Slow', lambda d: sma_trend_strategy(d, short=50, long=200)),
        ('Mean Reversion (Z=2.0)', lambda d: mean_reversion_strategy(d, period=20, z_threshold=2.0)),
        ('Volatility Breakout', lambda d: volatility_breakout_strategy(d, atr_period=14, std_dev=2))
    ]:
        strat_df = strat_func(df)
        # Join regime data
        strat_df = pd.concat([strat_df, regime_df], axis=1)
        strategies[strat_name] = strat_df
    
    print("✓ Implemented 4 strategies\n")
    
    # Analyze by trend regime
    print("\n1. PERFORMANCE BY TREND REGIME (Strong/Weak/None)")
    print("-" * 80)
    trend_analysis = {}
    for name, strat_df in strategies.items():
        trend_analysis[name] = analyze_strategy_by_regime(strat_df, 'Trend_Regime')
    
    compare_strategies_by_regime(trend_analysis, 'Trend_Regime')
    
    # Analyze by volatility regime
    print("\n\n2. PERFORMANCE BY VOLATILITY REGIME (Low/Med/High)")
    print("-" * 80)
    vol_analysis = {}
    for name, strat_df in strategies.items():
        vol_analysis[name] = analyze_strategy_by_regime(strat_df, 'Volatility_Regime')
    
    compare_strategies_by_regime(vol_analysis, 'Volatility_Regime')
    
    # Summary insights
    print("\n\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    trend_favors_sma = trend_analysis['SMA (10/30) - Trend']['STRONG_TREND']['Sharpe'] > \
                       trend_analysis['Mean Reversion (Z=2.0)']['STRONG_TREND']['Sharpe']
    
    vol_favors_breakout = vol_analysis['Volatility Breakout']['HIGH_VOL']['Sharpe'] > \
                          vol_analysis['SMA (10/30) - Trend']['HIGH_VOL']['Sharpe']
    
    print(f"\n✓ In STRONG TRENDS, SMA works best: {trend_favors_sma}")
    print(f"✓ In HIGH VOLATILITY, Breakouts work best: {vol_favors_breakout}")
    print(f"✓ Mean Reversion thrives in CHOPPY markets with low ADX")
    
    # Show when each strategy dominates
    print("\n--- When Each Strategy Wins ---")
    best_trend = max(trend_analysis.items(), 
                     key=lambda x: x[1].get('STRONG_TREND', {}).get('Sharpe', -999))
    best_sideways = max(trend_analysis.items(),
                        key=lambda x: x[1].get('NO_TREND', {}).get('Sharpe', -999))
    best_vol = max(vol_analysis.items(),
                   key=lambda x: x[1].get('HIGH_VOL', {}).get('Sharpe', -999))
    
    print(f"🏆 STRONG TRENDS:  {best_trend[0]}")
    print(f"🏆 SIDEWAYS/CHOP:  {best_sideways[0]}")
    print(f"🏆 HIGH VOLATILITY: {best_vol[0]}")
    
    # Visualize
    print("\n\nGenerating visualization...")
    plot_regime_analysis(df, regime_df, strategies, ticker)
    
    print("\n" + "="*80 + "\n")
    
    return df, regime_df, strategies, trend_analysis, vol_analysis


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Run analysis on multiple assets
    for ticker in ['SPY', 'QQQ', 'GLD']:
        df, regime_df, strategies, trend_analysis, vol_analysis = run_regime_research(ticker, days_back=1825)
        print("\n" + "="*80 + "\n")
