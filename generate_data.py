"""
Surveillance Alert Data Generator
Generates realistic synthetic surveillance data for training ML model
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

def generate_surveillance_alerts(n_samples=1000, true_violation_rate=0.15):
    """
    Generate synthetic surveillance alert data
    
    Args:
        n_samples: Total number of alerts to generate
        true_violation_rate: Proportion of actual violations (default 15%)
    
    Returns:
        DataFrame with alerts and features
    """
    n_violations = int(n_samples * true_violation_rate)
    n_false_positives = n_samples - n_violations
    
    alerts = []
    alert_id = 1
    
    # Generate TRUE VIOLATIONS (genuine market manipulation patterns)
    for _ in range(n_violations):
        alert = {
            'alert_id': f'ALT-{alert_id:05d}',
            'timestamp': datetime.now() - timedelta(days=random.randint(1, 90)),
            'trader_id': f'TR-{random.randint(1000, 9999)}',
            'symbol': random.choice(['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMC', 'GME', 'SPY']),
            'asset_class': random.choice(['Equity', 'Option']),
            
            # HIGH-RISK FEATURES (indicative of violations)
            'order_to_trade_ratio': np.random.uniform(15, 50),  # High = layering
            'trade_velocity': np.random.uniform(8, 25),  # orders per minute
            'price_deviation_pct': np.random.uniform(2.5, 8.0),  # % from VWAP
            'counterparty_concentration': np.random.uniform(0.6, 0.95),  # Wash trading
            'time_clustering_5min': np.random.randint(12, 40),  # orders in 5-min window
            'order_cancel_rate': np.random.uniform(0.7, 0.95),  # High cancellation
            'volume_spike': np.random.uniform(3.5, 15.0),  # Multiple of avg volume
            'spread_impact_bps': np.random.uniform(25, 100),  # Bid-ask spread impact
            'after_hours_trading': 1 if random.random() > 0.3 else 0,
            'cross_account_pattern': 1 if random.random() > 0.4 else 0,
            'previous_violations': random.randint(1, 5),
            'account_age_days': random.randint(10, 500),
            'avg_trade_size_usd': np.random.uniform(50000, 500000),
            'order_size_std': np.random.uniform(15000, 80000),
            'late_day_concentration': np.random.uniform(0.5, 0.9),
            'round_lot_pct': np.random.uniform(0.3, 0.7),
            'iceberg_order_flag': 1 if random.random() > 0.5 else 0,
            'momentum_following': np.random.uniform(0.6, 0.95),
            
            # LABEL
            'is_violation': 1
        }
        alerts.append(alert)
        alert_id += 1
    
    # Generate FALSE POSITIVES (legitimate trading flagged by rules)
    for _ in range(n_false_positives):
        alert = {
            'alert_id': f'ALT-{alert_id:05d}',
            'timestamp': datetime.now() - timedelta(days=random.randint(1, 90)),
            'trader_id': f'TR-{random.randint(1000, 9999)}',
            'symbol': random.choice(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']),
            'asset_class': random.choice(['Equity', 'Option']),
            
            # LOWER-RISK FEATURES (normal trading patterns)
            'order_to_trade_ratio': np.random.uniform(1, 12),
            'trade_velocity': np.random.uniform(0.5, 6),
            'price_deviation_pct': np.random.uniform(0.1, 2.0),
            'counterparty_concentration': np.random.uniform(0.1, 0.5),
            'time_clustering_5min': np.random.randint(1, 10),
            'order_cancel_rate': np.random.uniform(0.1, 0.6),
            'volume_spike': np.random.uniform(0.8, 3.0),
            'spread_impact_bps': np.random.uniform(1, 20),
            'after_hours_trading': 1 if random.random() > 0.8 else 0,
            'cross_account_pattern': 1 if random.random() > 0.9 else 0,
            'previous_violations': 0 if random.random() > 0.1 else random.randint(0, 1),
            'account_age_days': random.randint(180, 3000),
            'avg_trade_size_usd': np.random.uniform(5000, 150000),
            'order_size_std': np.random.uniform(1000, 25000),
            'late_day_concentration': np.random.uniform(0.1, 0.4),
            'round_lot_pct': np.random.uniform(0.6, 0.95),
            'iceberg_order_flag': 1 if random.random() > 0.8 else 0,
            'momentum_following': np.random.uniform(0.1, 0.5),
            
            # LABEL
            'is_violation': 0
        }
        alerts.append(alert)
        alert_id += 1
    
    # Convert to DataFrame and shuffle
    df = pd.DataFrame(alerts)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df


def save_training_data(df, output_path='data/surveillance_alerts.csv'):
    """Save generated data to CSV"""
    df.to_csv(output_path, index=False)
    print(f" Generated {len(df)} alerts")
    print(f"   - True violations: {df['is_violation'].sum()} ({df['is_violation'].mean()*100:.1f}%)")
    print(f"   - False positives: {(1-df['is_violation']).sum()} ({(1-df['is_violation'].mean())*100:.1f}%)")
    print(f"   - Saved to: {output_path}")
    return df


if __name__ == "__main__":
    df = generate_surveillance_alerts(n_samples=1000, true_violation_rate=0.15)
    save_training_data(df, 'data/surveillance_alerts.csv')
    print("\n Sample alerts:")
    print(df[['alert_id', 'symbol', 'order_to_trade_ratio', 'trade_velocity', 
              'counterparty_concentration', 'is_violation']].head(10))
