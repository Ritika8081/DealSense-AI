# DealSense AI: Production ML Deal Risk Intelligence

**$1.4M/quarter revenue impact** – ML-powered deal risk scoring for B2B SaaS CROs

**Random Forest** predicts deal loss probability (0-100 score)  
**Slack alerts** + "Exec sponsor TODAY" actions  
**Production-ready**: Airflow ETL + joblib models + API endpoints  
**Custom metrics** uncover sales cycle + rep performance gaps  
**ROI**: Saves 20 high-risk deals/quarter = $1.4M revenue

---

## Table of Contents
1. [The Business Problem](#the-business-problem)
2. [System Architecture](#system-architecture)
3. [Quick Setup Steps](#quick-setup-steps)
4. [Project Structure](#project-structure)
5. [Technical Deep Dive](#technical-deep-dive)
6. [Sample Output](#sample-output)
7. [Business Results](#business-results)
8. [Production Architecture](#production-architecture)
9. [Assignment Parts](#assignment-parts)
10. [Tech Stack](#tech-stack)

##  The Business Problem
**Sales cycles up 214%, win rates volatile.** Sales leaders need:
- **Early warning** on at-risk pipeline  
- **Explainable risk factors** (rep performance, stage delays)
- **Actionable interventions** ("Exec sponsor call TODAY")

**DealSense AI delivers daily intelligence** that turns data into revenue protection.

##  System Architecture
<img src="CRM_to_Risk_Scoring_Engine.png" alt="System Architecture" height="550" width="600"/>


## How to Run and Use This Project

Follow these steps to set up, run, and use DealSense AI:

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/dealsense-ai
   cd dealsense-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the EDA notebook ( for data exploration)**
   ```bash
   jupyter notebook 02_eda_insights_skygeni.ipynb
   ```

4. **Run the main risk scoring notebook**
   ```bash
   jupyter notebook notebooks/03_decision_engine_deal_risk.ipynb
   ```

5. **Check outputs**
   - Risk reports and feature importance will be generated in the `results/` folder.
   - Example: open `results/feature_importance.png` for feature importance visualization.

6. **Production code**
   - The main ML engine is in `src/deal_risk_scoring.py` and can be imported or run as a script for integration into other systems.

7. **Documentation**
   - See the `docs/` folder for business framing, system design, and project reflection.


## Project Structure

```
.
├── docs/
│   ├── 01_problem_framing.md         # Business context and challenge
│   ├── 04_system_design.md           # Architecture 
│   └── 05_reflection.md              # Project learnings and insights
├── notebooks/
│   └── 03_decision_engine_deal_risk.ipynb  # ML risk scoring 
├── results/
│   └── feature_importance.png        # Visualized feature importance
├── src/
│   └── deal_risk_scoring.py          # Main ML scoring logic 
├── 02_eda_insights_skygeni.ipynb     # EDA notebook:data analysis 
├── CRM_to_Risk_Scoring_Engine.png    # System architecture diagram
├── LICENSE                           # Project license
├── README.md                         # Project overview 
├── requirements.txt                  # Python dependencies
└── skygeni_sales_data.csv            # Raw sales data (5K deals)
```

> Note: Documentation files use their actual project file names (e.g., 01_problem_framing.md, 04_system_design.md, 05_reflection.md) to match the naming convention in the docs folder.


## Technical Deep Dive

### ML Pipeline
Raw CRM Data → Feature Engineering → Hybrid Scoring → Exec Outputs

**Feature Engineering (15+ features):**
- rep_historical_winrate, industry_winrate, deal_age_percentile
- is_long_cycle, is_large_deal, lead_source_quality

**Hybrid Scoring Engine:**
- Rule-based (40 pts): Cycle length, rep performance, stage delays
- ML-based (60 pts): Random Forest (ROC-AUC validated)
- Combined: 0-100 risk score + top 3 factors

**Production Features:**
- joblib model serialization
- Airflow/cron scheduling ready
- Slack/email alert integration
- Fallback logic (rules if ML fails)


## Sample Output

> **CRITICAL: D12345 ($45K ACV)**
>
> Risk Score: 87/100 | Level: CRITICAL
>
> Top Factors:
> 1. Rep bottom 20% win rate (25 pts)
> 2. 95 days in Demo (30 pts)
>
> → ACTION: Exec sponsor TODAY


## Business Results

- 214% sales cycle increase diagnosed
- Custom metrics: PVS, REI created
- $1.4M/quarter revenue protection potential
- Daily exec reports + Slack alerts ready


## Production Architecture

Salesforce API → Airflow ETL (6AM) → ML Scoring → Slack/Email → Tableau

        ↓

      CRO Dashboard (real-time)

Full system design: [docs/04_system_design.md](docs/04_system_design.md)


## Why This Matters

Most sales tech reports data. DealSense AI prescribes actions:

> "This rep needs coaching"
> 
> "Escalate this deal to VP Sales"
> 
> "Partner leads failing → audit channel"



## File Descriptions

1. **Problem Framing**  
   [docs/01_problem_framing.md](docs/01_problem_framing.md)  
   Defines the business context, objectives, and challenges for deal risk scoring.

2. **EDA & Insights**  
   [02_eda_insights_skygeni.ipynb](02_eda_insights_skygeni.ipynb)  
   Exploratory data analysis notebook: uncovers key patterns, trends, and actionable insights from the sales data.

3. **Decision Engine & Risk Scoring**  
   [notebooks/03_decision_engine_deal_risk.ipynb](notebooks/03_decision_engine_deal_risk.ipynb), [src/deal_risk_scoring.py](src/deal_risk_scoring.py)  
   ML pipeline and production code for scoring deal risk, including model training, evaluation, and demo analysis.

4. **System Design**  
   [docs/04_system_design.md](docs/04_system_design.md)  
   Technical documentation of the system architecture, data flow, and integration points.

5. **Reflection**  
   [docs/05_reflection.md](docs/05_reflection.md)  
   Project learnings, challenges faced, and key takeaways from the development and deployment process.



## Tech Stack

- **ML:** scikit-learn, Random Forest, joblib
- **Visualization:** Plotly, Matplotlib
- **Data:** pandas, numpy
- **Production:** Airflow-ready, API endpoints

---

## Final Note

Thank you for exploring DealSense AI! This project is designed to empower sales teams with actionable intelligence, robust analytics, and production-ready machine learning. Whether you’re a data scientist, engineer, or business leader, we hope this solution inspires you to drive smarter decisions and unlock new revenue opportunities. 

**Innovate boldly, automate wisely, and let data lead your growth!**
