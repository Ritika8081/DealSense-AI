
# Part 5 – Reflection

## Weakest Assumptions
1. **CRM data completeness** – Assumed stages/outcomes are accurate, but reps often "park" deals
2. **Static risk factors** – Assumed historical patterns persist, ignoring market shifts

## Production Break Points
1. **Stale CRM sync** → No new data = stale scores
2. **Data drift** → Model ROC-AUC drops below 0.7
3. **Rep gaming** → Risk scores manipulated via stage changes

## Next 1 Month Build
**Real-time API**: `/score_deal` endpoint + Slack bot for instant risk checks

## Least Confident
**ML model generalizes poorly** to new reps/regions (small test set). Would validate with holdout quarters.
