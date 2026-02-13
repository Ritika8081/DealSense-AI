"""
deal_risk_scoring.py
Production-ready deal risk scoring module for SkyGeni

This module provides:
1. Rule-based risk scoring with detailed factor analysis
2. ML-based probability prediction
3. Actionable risk report generation
4. Benchmark management

Usage:
    from deal_risk_scoring import DealRiskScorer
    
    scorer = DealRiskScorer()
    scorer.load_benchmarks('skygeni_sales_data.csv')
    
    risk_score, factors = scorer.calculate_risk(deal_dict)
    report = scorer.generate_report(deal_dict, risk_score, factors)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import joblib
from pathlib import Path

@dataclass
class RiskFactor:
    """Represents a single risk factor"""
    description: str
    points: int
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'

class DealRiskScorer:
    """
    Production-ready deal risk scoring system.
    
    Combines rule-based and ML-based approaches for comprehensive risk assessment.
    """
    
    def __init__(self, model=None):
        self.model = model
        self.benchmarks = {}
        self.feature_cols = []
        
    def load_benchmarks(self, data_path: str):
        """
        Calculate historical benchmarks from training data.
        
        Args:
            data_path: Path to CSV file with historical deals
        """
        df = pd.read_csv(data_path)
        df['created_date'] = pd.to_datetime(df['created_date'])
        df['closed_date'] = pd.to_datetime(df['closed_date'])
        
        # Calculate benchmarks
        self.benchmarks = {
            'rep_win_rates': df.groupby('sales_rep_id')['outcome'].apply(
                lambda x: (x == 'Won').mean() * 100
            ).to_dict(),
            
            'industry_win_rates': df.groupby('industry')['outcome'].apply(
                lambda x: (x == 'Won').mean() * 100
            ).to_dict(),
            
            'lead_source_win_rates': df.groupby('lead_source')['outcome'].apply(
                lambda x: (x == 'Won').mean() * 100
            ).to_dict(),
            
            'region_win_rates': df.groupby('region')['outcome'].apply(
                lambda x: (x == 'Won').mean() * 100
            ).to_dict(),
            
            'avg_sales_cycle': df['sales_cycle_days'].mean(),
            'avg_deal_size': df['deal_amount'].mean(),
            'overall_win_rate': (df['outcome'] == 'Won').mean() * 100,
            
            # Percentile thresholds
            'rep_win_rate_q20': df.groupby('sales_rep_id')['outcome'].apply(
                lambda x: (x == 'Won').mean() * 100
            ).quantile(0.2),
            'rep_win_rate_q80': df.groupby('sales_rep_id')['outcome'].apply(
                lambda x: (x == 'Won').mean() * 100
            ).quantile(0.8),
        }
        
        print(f"✅ Benchmarks loaded from {data_path}")
        print(f"   Average sales cycle: {self.benchmarks['avg_sales_cycle']:.0f} days")
        print(f"   Overall win rate: {self.benchmarks['overall_win_rate']:.1f}%")
    
    def calculate_rule_based_risk(self, deal: Dict) -> Tuple[int, List[RiskFactor]]:
        """
        Calculate rule-based risk score (0-100) with detailed factors.
        
        Args:
            deal: Dictionary with deal attributes (sales_cycle_days, sales_rep_id, etc.)
            
        Returns:
            Tuple of (risk_score, list_of_risk_factors)
        """
        if not self.benchmarks:
            raise ValueError("Benchmarks not loaded. Call load_benchmarks() first.")
        
        risk_score = 0
        risk_factors = []
        
        # Factor 1: Sales Cycle Length (30 points max)
        cycle_days = deal.get('sales_cycle_days', 0)
        avg_cycle = self.benchmarks['avg_sales_cycle']
        
        if cycle_days > 90:
            points = 30
            risk_score += points
            risk_factors.append(RiskFactor(
                f"Deal stuck for 90+ days (critical threshold)",
                points, 'CRITICAL'
            ))
        elif cycle_days > 70:
            points = 20
            risk_score += points
            risk_factors.append(RiskFactor(
                f"Slow cycle: {cycle_days} days (avg: {avg_cycle:.0f})",
                points, 'HIGH'
            ))
        elif cycle_days > avg_cycle:
            points = 10
            risk_score += points
            risk_factors.append(RiskFactor(
                f"Above-average cycle: {cycle_days} days",
                points, 'MEDIUM'
            ))
        
        # Factor 2: Rep Historical Performance (25 points max)
        rep_id = deal.get('sales_rep_id')
        if rep_id and rep_id in self.benchmarks['rep_win_rates']:
            rep_wr = self.benchmarks['rep_win_rates'][rep_id]
            q20 = self.benchmarks['rep_win_rate_q20']
            
            if rep_wr < q20:
                points = 25
                risk_score += points
                risk_factors.append(RiskFactor(
                    f"Underperforming rep: {rep_wr:.1f}% win rate (bottom 20%)",
                    points, 'HIGH'
                ))
            elif rep_wr < self.benchmarks['overall_win_rate']:
                points = 15
                risk_score += points
                risk_factors.append(RiskFactor(
                    f"Below-average rep: {rep_wr:.1f}% win rate",
                    points, 'MEDIUM'
                ))
        
        # Factor 3: Industry Win Rate (15 points)
        industry = deal.get('industry')
        if industry and industry in self.benchmarks['industry_win_rates']:
            ind_wr = self.benchmarks['industry_win_rates'][industry]
            if ind_wr < 44.5:
                points = 15
                risk_score += points
                risk_factors.append(RiskFactor(
                    f"{industry} struggling: {ind_wr:.1f}% win rate",
                    points, 'MEDIUM'
                ))
        
        # Factor 4: Lead Source Quality (15 points)
        lead_source = deal.get('lead_source')
        if lead_source and lead_source in self.benchmarks['lead_source_win_rates']:
            lead_wr = self.benchmarks['lead_source_win_rates'][lead_source]
            if lead_source == 'Partner' or lead_wr < 45:
                points = 15
                risk_score += points
                risk_factors.append(RiskFactor(
                    f"Low-quality lead source: {lead_source} ({lead_wr:.1f}% win rate)",
                    points, 'MEDIUM'
                ))
        
        # Factor 5: Deal Stage Duration (10 points)
        stage = deal.get('deal_stage', '')
        if stage in ['Demo', 'Proposal'] and cycle_days > 40:
            points = 10
            risk_score += points
            risk_factors.append(RiskFactor(
                f"Stuck in {stage} stage for {cycle_days} days",
                points, 'MEDIUM'
            ))
        
        # Factor 6: Deal Size Extremes (5 points)
        deal_amount = deal.get('deal_amount', 0)
        if deal_amount > 50000:
            points = 5
            risk_score += points
            risk_factors.append(RiskFactor(
                f"Large deal (${deal_amount:,}) - higher scrutiny risk",
                points, 'LOW'
            ))
        
        # Factor 7: Region Performance (5 points)
        region = deal.get('region')
        if region and region in self.benchmarks['region_win_rates']:
            region_wr = self.benchmarks['region_win_rates'][region]
            if region_wr < 45:
                points = 5
                risk_score += points
                risk_factors.append(RiskFactor(
                    f"{region} region below average: {region_wr:.1f}%",
                    points, 'LOW'
                ))
        
        # Sort by points descending
        risk_factors.sort(key=lambda x: x.points, reverse=True)
        
        return min(risk_score, 100), risk_factors
    
    def classify_risk_level(self, risk_score: int) -> str:
        """Convert numeric risk score to categorical level"""
        if risk_score >= 75:
            return 'CRITICAL'
        elif risk_score >= 55:
            return 'HIGH'
        elif risk_score >= 30:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_report(
        self, 
        deal: Dict, 
        risk_score: int, 
        risk_factors: List[RiskFactor],
        top_n: int = 3
    ) -> str:
        """
        Generate executive-ready risk report for a deal.
        
        Args:
            deal: Deal dictionary
            risk_score: Computed risk score (0-100)
            risk_factors: List of RiskFactor objects
            top_n: Number of top factors to include
            
        Returns:
            Formatted report string
        """
        risk_level = self.classify_risk_level(risk_score)
        
        # Generate actions based on risk factors
        actions = []
        for factor in risk_factors[:top_n]:
            desc_lower = factor.description.lower()
            if 'stuck' in desc_lower or 'slow' in desc_lower:
                actions.append("→ Schedule executive sponsor meeting within 48 hours")
            if 'rep' in desc_lower and 'underperform' in desc_lower:
                actions.append("→ Assign senior sales leader to shadow/coach on this deal")
            if 'demo' in desc_lower or 'proposal' in desc_lower:
                actions.append("→ Engage solutions engineer for technical deep-dive")
            if 'partner' in desc_lower:
                actions.append("→ Verify partner lead quality and buyer authority")
            if 'large deal' in desc_lower:
                actions.append("→ Involve VP Sales for C-level relationship building")
        
        # Remove duplicates, keep first 3
        actions = list(dict.fromkeys(actions))[:3]
        if not actions:
            actions = ["→ Monitor closely, no immediate action required"]
        
        # Build report
        report = f"""
{'='*70}
🚨 DEAL RISK ALERT: {deal.get('deal_id', 'N/A')} | {risk_level} RISK
{'='*70}

Account: [Company Name]  |  ACV: ${deal.get('deal_amount', 0):,}  |  Rep: {deal.get('sales_rep_id', 'N/A')}
Industry: {deal.get('industry', 'N/A')}  |  Region: {deal.get('region', 'N/A')}  |  Stage: {deal.get('deal_stage', 'N/A')}
Days in Pipeline: {deal.get('sales_cycle_days', 0)}  |  Risk Score: {risk_score}/100

⚠️  TOP RISK FACTORS:
"""
        
        for i, factor in enumerate(risk_factors[:top_n], 1):
            report += f"   {i}. {factor.description} ({factor.points} pts - {factor.severity})\n"
        
        report += f"\n💡 RECOMMENDED ACTIONS:\n"
        for action in actions:
            report += f"   {action}\n"
        
        # Context
        report += f"\n📊 CONTEXT:\n"
        rep_id = deal.get('sales_rep_id')
        if rep_id and rep_id in self.benchmarks['rep_win_rates']:
            rep_wr = self.benchmarks['rep_win_rates'][rep_id]
            report += f"   - Rep {rep_id} historical win rate: {rep_wr:.1f}%\n"
        
        industry = deal.get('industry')
        if industry and industry in self.benchmarks['industry_win_rates']:
            ind_wr = self.benchmarks['industry_win_rates'][industry]
            report += f"   - {industry} industry win rate: {ind_wr:.1f}%\n"
        
        report += f"   - Average sales cycle: {self.benchmarks['avg_sales_cycle']:.0f} days "
        report += f"(this deal: {deal.get('sales_cycle_days', 0)} days)\n"
        report += "="*70
        
        return report
    
    # ML Model Methods
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for ML model.
        
        Args:
            df: DataFrame with raw deal data
            
        Returns:
            DataFrame with engineered features
        """
        df_ml = df.copy()
        
        # Add historical performance features
        df_ml['rep_historical_win_rate'] = df_ml['sales_rep_id'].map(
            self.benchmarks.get('rep_win_rates', {})
        )
        df_ml['industry_win_rate'] = df_ml['industry'].map(
            self.benchmarks.get('industry_win_rates', {})
        )
        df_ml['lead_source_win_rate'] = df_ml['lead_source'].map(
            self.benchmarks.get('lead_source_win_rates', {})
        )
        df_ml['region_win_rate'] = df_ml['region'].map(
            self.benchmarks.get('region_win_rates', {})
        )
        
        # Deal characteristics
        df_ml['deal_size_percentile'] = df_ml['deal_amount'].rank(pct=True)
        df_ml['sales_cycle_vs_avg'] = df_ml['sales_cycle_days'] / self.benchmarks.get('avg_sales_cycle', 63)
        df_ml['is_large_deal'] = (df_ml['deal_amount'] > 30000).astype(int)
        df_ml['is_long_cycle'] = (df_ml['sales_cycle_days'] > 70).astype(int)
        
        # Temporal features
        if 'closed_date' in df_ml.columns:
            df_ml['closed_date'] = pd.to_datetime(df_ml['closed_date'])
            df_ml['quarter'] = df_ml['closed_date'].dt.quarter
            df_ml['is_q4'] = (df_ml['quarter'] == 4).astype(int)
        
        # Encode categoricals
        categorical_cols = ['industry', 'region', 'product_type', 'lead_source', 'deal_stage']
        df_ml_encoded = pd.get_dummies(df_ml, columns=categorical_cols, drop_first=True)
        
        return df_ml_encoded
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train ML model"""
        from sklearn.ensemble import RandomForestClassifier
        
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X, y)
        self.feature_cols = X.columns.tolist()
        print(f"✅ Model trained on {len(X)} samples with {len(self.feature_cols)} features")
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict loss probability"""
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]
    
    def save_model(self, path: str):
        """Save model and benchmarks"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'benchmarks': self.benchmarks,
            'feature_cols': self.feature_cols
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and benchmarks"""
        data = joblib.load(path)
        self.model = data['model']
        self.benchmarks = data['benchmarks']
        self.feature_cols = data.get('feature_cols', [])
        print(f"✅ Model loaded from {path}")

# Utility functions
def generate_daily_report(
    deals_df: pd.DataFrame,
    scorer: DealRiskScorer,
    date_str: str,
    top_n: int = 5
) -> str:
    """
    Generate daily risk report for all open deals.
    
    Args:
        deals_df: DataFrame with deal data
        scorer: Trained DealRiskScorer instance
        date_str: Date string for report header
        top_n: Number of top risky deals to highlight
        
    Returns:
        Formatted daily report
    """
    # Calculate risk scores for all deals
    deals_df['risk_score'] = 0
    deals_df['risk_level'] = 'LOW'
    
    for idx, row in deals_df.iterrows():
        deal_dict = row.to_dict()
        score, factors = scorer.calculate_rule_based_risk(deal_dict)
        deals_df.at[idx, 'risk_score'] = score
        deals_df.at[idx, 'risk_level'] = scorer.classify_risk_level(score)
    
    # Generate summary report
    critical_count = len(deals_df[deals_df['risk_level'] == 'CRITICAL'])
    high_count = len(deals_df[deals_df['risk_level'] == 'HIGH'])
    medium_count = len(deals_df[deals_df['risk_level'] == 'MEDIUM'])
    low_count = len(deals_df[deals_df['risk_level'] == 'LOW'])
    
    critical_acv = deals_df[deals_df['risk_level'] == 'CRITICAL']['deal_amount'].sum()
    
    report = f"""
{'='*80}
📊 DAILY DEAL RISK REPORT - {date_str}
{'='*80}

EXECUTIVE SUMMARY:
──────────────────────────────────────────────────────────────────────────────
Open Deals: {len(deals_df)}
Total Pipeline ACV: ${deals_df['deal_amount'].sum():,}

Risk Distribution:
  🔴 CRITICAL: {critical_count} deals (${critical_acv:,} at risk)
  🟠 HIGH:     {high_count} deals
  🟡 MEDIUM:   {medium_count} deals
  🟢 LOW:      {low_count} deals

{'='*80}
🚨 TOP {top_n} DEALS REQUIRING IMMEDIATE ATTENTION
{'='*80}
"""
    
    top_deals = deals_df.nlargest(top_n, 'risk_score')
    
    for i, (idx, deal) in enumerate(top_deals.iterrows(), 1):
        deal_dict = deal.to_dict()
        score, factors = scorer.calculate_rule_based_risk(deal_dict)
        
        report += f"\nDeal #{i}: {deal['deal_id']}\n"
        report += f"{'─'*76}\n"
        report += f"ACV: ${deal['deal_amount']:,}  |  Rep: {deal['sales_rep_id']}  "
        report += f"|  Stage: {deal['deal_stage']}\n"
        report += f"Risk Score: {score}/100  |  Level: {scorer.classify_risk_level(score)}\n"
        report += f"Top 3 Risk Factors:\n"
        for j, factor in enumerate(factors[:3], 1):
            report += f"  {j}. {factor.description} ({factor.points} pts)\n"
    
    report += "\n" + "="*80
    
    return report

if __name__ == "__main__":
    # Example usage
    print("SkyGeni Deal Risk Scoring Module")
    print("="*50)
    
    # Initialize scorer
    scorer = DealRiskScorer()
    
    # Load benchmarks
    scorer.load_benchmarks('skygeni_sales_data.csv')
    
    # Example deal
    example_deal = {
        'deal_id': 'D12345',
        'sales_rep_id': 'rep_22',
        'industry': 'EdTech',
        'region': 'APAC',
        'product_type': 'Enterprise',
        'lead_source': 'Partner',
        'deal_stage': 'Demo',
        'deal_amount': 45000,
        'sales_cycle_days': 95
    }
    
    # Calculate risk
    risk_score, risk_factors = scorer.calculate_rule_based_risk(example_deal)
    
    # Generate report
    report = scorer.generate_report(example_deal, risk_score, risk_factors)
    print("\n" + report)
