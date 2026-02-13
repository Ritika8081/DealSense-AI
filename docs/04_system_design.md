# SkyGeni Deal Risk Decision Engine: System Design

## Overview
The SkyGeni Deal Risk Decision Engine is designed to provide daily, actionable risk insights for B2B SaaS sales teams. It combines rule-based and machine learning (ML) models to score deals, generate risk reports, and highlight key risk factors for executive decision-making.

## 🎯 Data Flow
```mermaid
flowchart TD
    CRM[CRM (Salesforce/HubSpot)] --> ETL[Daily ETL Job (6AM PST)]
    ETL --> FeatureStore[Feature Store]
    FeatureStore --> RiskEngine[Risk Scoring Engine\n(Rule-based + ML)]
    RiskEngine --> Reports[Risk Reports + Alerts]
    Reports --> Channels[Email/Slack/Tableau]
```

## 🏗️ Architecture Components
See below for detailed architecture.

## ⏰ Scheduling
- **Daily 6AM PST** via Airflow/cron
- **Real-time** scoring on deal stage changes (future)
- **SLA:** Reports delivered by 7AM PST

## 🚨 Example Alerts & Insights
🚨 **CRITICAL ALERT (3 deals @ risk):**
- D12345 (ACV $45k) - Rep bottom 20% win rate, 95 days in Demo
  → ACTION: Executive sponsor call TODAY
- D67890 (ACV $28k) - Partner lead + Proposal stage 60+ days
  → ACTION: Verify buyer authority + SE review

📊 **DAILY SUMMARY:** 12 high-risk deals ($1.4M ACV @ risk)

## ⚠️ Failure Cases & Mitigations
1. **No CRM data sync** → Send "data unavailable" alert
2. **ML model drift** → Auto-fallback to rule-based scoring
3. **High-risk deal volume spike** → Escalate to CRO dashboard
4. **Data quality issues** → Flag deals with >3 missing fields

## Architecture
- **Data Ingestion:**
  - Source: CRM export (CSV or API integration)
  - Automated daily data pull (future integration)
- **Feature Engineering:**
  - Custom feature generation (e.g., deal age, activity counts, engagement metrics)
  - Handled in `src/deal_risk_scoring.py`
- **Risk Scoring:**
  - Rule-based scoring for transparency and quick wins
  - ML-based scoring (Random Forest) for predictive accuracy
- **Reporting:**
  - Daily risk report CSVs
  - Feature importance plots
  - Executive summaries
- **Output:**
  - Results saved to `results/` folder
  - Ready for dashboard or presentation integration

## Integration & Automation
- **Current:** Manual notebook execution
- **Future:**
  - Schedule as a daily job (e.g., with Airflow, cron, or cloud scheduler)
  - Integrate with CRM via API for real-time scoring
  - Automated email or dashboard delivery of risk reports

## Monitoring & Maintenance
- Track model performance (ROC-AUC, precision/recall)
- Monitor data drift and retrain ML model as needed
- Log daily outputs and errors for auditability

## Security & Compliance
- Handle sensitive sales data securely
- Ensure compliance with company data policies

---

This system is designed for extensibility, automation, and executive usability, supporting SkyGeni's goal of data-driven sales excellence.
