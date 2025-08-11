# Interference & Spillover Detection Tools

**Objective**  
Identify and quantify spillover effects in marketing campaigns (e.g., opt-out group exposed via in-store signage).

**Scope / Acceptance**  
- Exposure mapping for partial interference designs.  
- Spillover effect estimation in networked or clustered data.

**Tasks**  
1. Implement exposure mapping by store/geofence/social ties.  
2. Model outcomes as a function of own-treatment and neighbor-treatment exposure.  
3. Support two-stage randomization inference for interference.  
4. Diagnostics: cluster-level exposure balance plots.

**Test Data**  
- Synthetic cluster network with assigned treatments and controlled spillover.  
- Loyalty program data with store visit logs.

**Tests**  
- Detect simulated spillover ≥ 0.2 with p < 0.05.  
- No false detection when spillover = 0.

**KPIs**  
- Detection power ≥ 80% for medium spillover effects in 1k clusters.
