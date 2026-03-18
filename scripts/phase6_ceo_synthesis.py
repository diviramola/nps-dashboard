"""
Phase 6: CEO Synthesis — Leadership Deliverable
=================================================
Produces the final executive report distilling all phases (0-5) into
actionable insights for Wiom's leadership team.

Structure:
  1. Executive Summary (1 page)
  2. Top 5 Driver Deep-Dives (1 page each)
  3. Segmentation View
  4. Industry Expert Independent View
  5. Monitoring Framework
  6. Appendices (data quality, methodology)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os, sys

OUT_DIR = r'C:\Users\nikhi\wiom-nps-analysis\output'
DATA_DIR = r'C:\Users\nikhi\wiom-nps-analysis\data'
os.makedirs(OUT_DIR, exist_ok=True)

report_lines = []
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def R(line=''):
    report_lines.append(line)
    print(line)

def save_report():
    path = os.path.join(OUT_DIR, 'phase6_ceo_synthesis.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  Saved: {path}")

# Load data
df = pd.read_csv(os.path.join(DATA_DIR, 'nps_with_risk_scores.csv'), low_memory=False)

# ══════════════════════════════════════════════════════════════════════════
# Compute key metrics from the data
# ══════════════════════════════════════════════════════════════════════════

total_n = len(df)
nps_col = 'nps_score' if 'nps_score' in df.columns else 'NPS'
nps_group_col = 'nps_group' if 'nps_group' in df.columns else 'NPS Group'
churn_col = 'is_churned' if 'is_churned' in df.columns else 'churn_flag'

# Find the right columns — exact match preferred over partial
for c in df.columns:
    if c.lower() in ('nps', 'nps_score'):
        nps_col = c
    if c.lower() == 'nps_group':
        nps_group_col = c
    if c.lower() in ('is_churned', 'churn_flag', 'churn_label'):
        churn_col = c

nps_scores = pd.to_numeric(df[nps_col], errors='coerce').dropna()
mean_nps = nps_scores.mean()

# NPS calculation (% Promoter - % Detractor) — exclude NaN rows
if nps_group_col in df.columns:
    grp_valid = df[nps_group_col].dropna()
    grp = grp_valid.astype(str).str.strip()
    total_valid = len(grp)
    pct_promoter = (grp.str.lower() == 'promoter').sum() / total_valid * 100
    pct_detractor = (grp.str.lower() == 'detractor').sum() / total_valid * 100
    pct_passive = (grp.str.lower() == 'passive').sum() / total_valid * 100
    nps_metric = pct_promoter - pct_detractor
else:
    pct_promoter = (nps_scores >= 9).mean() * 100
    pct_detractor = (nps_scores <= 6).mean() * 100
    pct_passive = ((nps_scores >= 7) & (nps_scores <= 8)).mean() * 100
    nps_metric = pct_promoter - pct_detractor

# Churn
if churn_col in df.columns:
    churn_vals = df[churn_col]
    if churn_vals.dtype == 'object':
        churn_rate = (churn_vals.astype(str).str.strip().str.lower() == 'churn').mean() * 100
    else:
        churn_rate = churn_vals.mean() * 100
else:
    churn_rate = 23.4  # from Phase 4

# Sprint trend
sprint_col = None
for c in df.columns:
    if 'sprint' in c.lower() and ('id' in c.lower() or 'num' in c.lower() or c.lower() == 'sprint'):
        sprint_col = c
        break

sprint_nps = {}
if sprint_col:
    for sp, grp_df in df.groupby(sprint_col):
        sp_scores = pd.to_numeric(grp_df[nps_col], errors='coerce').dropna()
        if len(sp_scores) >= 30:
            sprint_nps[sp] = sp_scores.mean()

# Tenure breakdown
tenure_col = None
for c in df.columns:
    if 'tenure' in c.lower() and 'excel' in c.lower():
        tenure_col = c
        break
if not tenure_col:
    for c in df.columns:
        if c.lower() in ('tenure', 'tenure_bucket'):
            tenure_col = c
            break

tenure_stats = {}
if tenure_col:
    for t, t_df in df.groupby(tenure_col):
        t_scores = pd.to_numeric(t_df[nps_col], errors='coerce').dropna()
        t_churn = 0
        if churn_col in t_df.columns:
            cv = t_df[churn_col]
            if cv.dtype == 'object':
                t_churn = (cv.astype(str).str.strip().str.lower() == 'churn').mean() * 100
            else:
                t_churn = cv.mean() * 100
        tenure_stats[str(t)] = {
            'n': len(t_df),
            'mean_nps': t_scores.mean() if len(t_scores) > 0 else 0,
            'churn': t_churn
        }

# City breakdown
city_col = None
for c in df.columns:
    if 'city' in c.lower() and 'core' in c.lower():
        city_col = c
        break
if not city_col:
    for c in df.columns:
        if c.lower() in ('city', 'city_core'):
            city_col = c
            break

city_stats = {}
if city_col:
    for ct, ct_df in df.groupby(city_col):
        ct_scores = pd.to_numeric(ct_df[nps_col], errors='coerce').dropna()
        ct_churn = 0
        if churn_col in ct_df.columns:
            cv = ct_df[churn_col]
            if cv.dtype == 'object':
                ct_churn = (cv.str.strip().str.lower() == 'churn').mean() * 100
            else:
                ct_churn = cv.mean() * 100
        if len(ct_df) >= 50:
            city_stats[str(ct)] = {
                'n': len(ct_df),
                'mean_nps': ct_scores.mean() if len(ct_scores) > 0 else 0,
                'churn': ct_churn
            }

# Risk score breakdown
risk_col = None
for c in df.columns:
    if c.lower() == 'risk_category':
        risk_col = c
        break
if not risk_col:
    for c in df.columns:
        if 'risk' in c.lower() and 'cat' in c.lower():
            risk_col = c
            break

risk_stats = {}
if risk_col:
    for r, r_df in df.groupby(risk_col):
        risk_stats[str(r)] = {'n': len(r_df), 'pct': len(r_df)/total_n*100}

# Segment breakdown
seg_col = None
for c in df.columns:
    if 'segment' in c.lower() or 'cluster' in c.lower():
        seg_col = c
        break

seg_stats = {}
if seg_col:
    for s, s_df in df.groupby(seg_col):
        s_scores = pd.to_numeric(s_df[nps_col], errors='coerce').dropna()
        s_churn = 0
        if churn_col in s_df.columns:
            cv = s_df[churn_col]
            if cv.dtype == 'object':
                s_churn = (cv.str.strip().str.lower() == 'churn').mean() * 100
            else:
                s_churn = cv.mean() * 100
        seg_stats[str(s)] = {
            'n': len(s_df),
            'mean_nps': s_scores.mean() if len(s_scores) > 0 else 0,
            'churn': s_churn
        }


# ══════════════════════════════════════════════════════════════════════════
# BEGIN REPORT
# ══════════════════════════════════════════════════════════════════════════

R('=' * 76)
R('  WIOM NPS DRIVER ANALYSIS')
R('  CEO SYNTHESIS & LEADERSHIP DELIVERABLE')
R('=' * 76)
R(f'  Prepared: {datetime.now().strftime("%B %d, %Y")}')
R(f'  Data Period: July 2025 (Sprint 1) through January 2026 (Sprint 13)')
R(f'  Sample: {total_n:,} NPS respondents | 5,488 free-text comments (42%)')
R(f'  Methodology: Triangulated qualitative (NLP/CX), quantitative (KDA/ML),')
R(f'  and independent industry expert assessment')
R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: EXECUTIVE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  1. EXECUTIVE SUMMARY')
R('=' * 76)
R()
R(f'  HEADLINE: Wiom\'s NPS problem is NOT primarily a network problem.')
R(f'  It is a HUMAN DELIVERY and PROCESS problem.')
R()
R(f'  THE NUMBERS')
R(f'  +-----------------------+-----------+')
R(f'  | NPS Score (net)       | {nps_metric:+.1f}      |')
R(f'  | Mean Score (0-10)     | {mean_nps:.1f}        |')
R(f'  | Promoters (9-10)      | {pct_promoter:.1f}%      |')
R(f'  | Passives (7-8)        | {pct_passive:.1f}%       |')
R(f'  | Detractors (0-6)      | {pct_detractor:.1f}%      |')
R(f'  | Overall Churn Rate    | {churn_rate:.1f}%      |')
R(f'  | Promoter Churn Rate   | 20.2%      |')
R(f'  +-----------------------+-----------+')
R()
R('  KEY FINDING: NPS scores do NOT predict churn beyond what operational')
R('  data already captures (AUC +0.0001 lift). The value of NPS is in the')
R('  COMMENTS (qualitative diagnosis), not the SCORE (quantitative prediction).')
R()

# Sprint trend
if sprint_nps:
    R('  NPS TREND BY SPRINT')
    def sprint_sort_key(item):
        s = str(item[0])
        # Extract numeric part: "Sprint 1" -> 1, "1" -> 1
        import re
        m = re.search(r'(\d+)', s)
        return int(m.group(1)) if m else 0
    sorted_sprints = sorted(sprint_nps.items(), key=sprint_sort_key)
    R('  Sprint  |  Avg NPS  |  Trend')
    R('  --------+-----------+--------')
    prev = None
    for sp, avg in sorted_sprints:
        trend = ''
        if prev is not None:
            diff = avg - prev
            if diff > 0.3:
                trend = f'  +{diff:.1f}'
            elif diff < -0.3:
                trend = f'  {diff:.1f}'
            else:
                trend = '  ~flat'
        R(f'  {str(sp):8s}|   {avg:.1f}     | {trend}')
        prev = avg
    R()

R('  TOP 5 DRIVERS (Triangulated, Ranked by Impact x Actionability)')
R()
R('  Rank | Driver                       | Confidence | What To Do')
R('  -----+------------------------------+------------+---------------------------')
R('    1  | Partner/Rohit Quality         | HIGH       | Quality tiering + training')
R('    2  | Support & Complaint Handling  | HIGH       | Service recovery SOP')
R('    3  | Network Stability             | MED-HIGH   | Device-level monitoring')
R('    4  | 28-Day Billing Cycle          | MED-HIGH   | Move to 30-day cycles')
R('    5  | Price/Value Perception        | MED-HIGH   | Protect affordability moat')
R()
R('  THE ONE SENTENCE FOR THE BOARD:')
R('  "The Rohit (partner) IS Wiom for the customer. Improving partner quality')
R('  and complaint handling will move NPS more than any network investment,')
R('  and eliminating the 28-day billing cycle removes a self-inflicted wound."')
R()

# NPS-Churn linkage
R('  NPS-CHURN LINKAGE')
R('  The churn model (AUC 0.977) identifies at-risk customers using')
R('  operational data alone. NPS adds zero incremental predictive power.')
R()
R('  What this means for leadership:')
R('  - Operational dashboards are SUFFICIENT for churn prediction')
R('  - NPS is useful for diagnosing WHY customers are unhappy (comments)')
R('  - NPS is NOT useful for predicting WHO will churn (score)')
R('  - Focus NPS investment on: closed-loop follow-up, comment analysis,')
R('    and theme tracking — NOT on score-based prediction models')
R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: DRIVER #1 DEEP DIVE — PARTNER QUALITY
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  2. DRIVER DEEP-DIVE #1: PARTNER/ROHIT QUALITY')
R('     "The Partner IS the Brand"')
R('=' * 76)
R()
R('  EVIDENCE STRENGTH: HIGH (all 3 perspectives converge)')
R()
R('  QUANTITATIVE')
R('  - partner_avg_nps is #1 KDA consensus driver (4/4 statistical methods)')
R('  - 100% sprint stability (appears in every sprint tested)')
R('  - partner_churn_rate is #2 churn predictor (11.4% feature importance)')
R('  - partner_sla_compliance, partner_fcr_rate in extended top-20')
R('  - Partner quality explains more NPS variance than any network metric')
R()
R('  QUALITATIVE')
R('  - technician_partner_bad theme: 70.6% Detractors, 23.5% churn rate')
R('  - complaint_resolution_bad: 85.1% Detractors, NPS 2.4')
R('  - Customer voices:')
R('    "Technician nahi aate" (technician doesn\'t come)')
R('    "Rohit ne bola 100 Mbps milega, magar 5-6 hi aati hai"')
R('    (Rohit promised 100 Mbps, but only getting 5-6)')
R('    "Ek hafte se complaint hai, koi dhyan nahi deta"')
R('    (Complaint for a week, nobody pays attention)')
R()
R('  INDUSTRY EXPERT VIEW')
R('  - Independent finding #3: "The Partner IS the Brand"')
R('  - Partner quality variance > geography variance')
R('  - "A bad partner = bad Wiom, regardless of network quality"')
R('  - This is a known pattern in last-mile ISPs: the field tech shapes')
R('    the ENTIRE perception of the service')
R()
R('  OPERATIONAL LEVER')
R('  Snowflake: partner_details_log (partner status), PARTNER_INFLUX_SUMMARY')
R('  (partner uptime), service_ticket_model (partner SLA), t_router_user_mapping')
R('  (partner customer portfolio)')
R()
R('  RECOMMENDED ACTION')
R('  1. Build Partner Quality Composite Score:')
R('     - Partner uptime % (30-day avg)')
R('     - Median install TAT')
R('     - Ticket SLA compliance rate')
R('     - Customer churn rate in portfolio')
R('     - Average NPS of their customers')
R('  2. Tier partners A/B/C based on composite score')
R('  3. A-tier: higher commissions, priority install allocation')
R('  4. C-tier: mandatory training, 90-day improvement window, supervised')
R('  5. Auto-escalate any complaint from C-tier partner customers')
R()
R('  EXPECTED IMPACT: Moving bottom-quartile partners to median could shift')
R('  15-20% of their Detractors toward Passive/Promoter. Estimated NPS')
R('  improvement: +3 to +5 points within 2 sprints of implementation.')
R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: DRIVER #2 DEEP DIVE — SUPPORT & COMPLAINT HANDLING
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  3. DRIVER DEEP-DIVE #2: SUPPORT EFFORT & COMPLAINT HANDLING')
R('     "How you handle problems matters more than problem frequency"')
R('=' * 76)
R()
R('  EVIDENCE STRENGTH: HIGH (all 3 perspectives converge)')
R()
R('  QUANTITATIVE')
R('  - tickets_before_3m: #4 KDA consensus driver (3/4 methods)')
R('  - AVG_TIMES_REOPENED: 47.7% higher for Detractors vs Promoters')
R('  - TOTAL_IVR_CALLS: 22.9% higher for Detractors')
R('  - MAX_TICKETS_SAME_ISSUE: 26.8% higher for Detractors')
R('  - support_effort_index: significant in regression models')
R()
R('  QUALITATIVE')
R('  - complaint_resolution_bad: 161 comments, 85.1% Detractors, NPS 2.4')
R('  - complaint_resolution_good: 10 comments, NPS 9.2 (a 6.8 point gap!)')
R('  - 28.9% of commenting Detractors cite complaint handling')
R('  - The SERVICE RECOVERY PARADOX is real in this data:')
R('    Good resolution of a problem creates a STRONGER promoter than')
R('    never having a problem at all')
R()
R('  INDUSTRY EXPERT VIEW')
R('  - Breakage Point #3: "First complaint resolution failure creates')
R('    permanent distrust"')
R('  - "Each additional call doubles churn probability"')
R('  - FCR (first contact resolution) is the gold standard metric')
R()
R('  CUSTOMER VOICES')
R('  - "Ek hafte se complaint hai, koi dhyan nahi deta"')
R('    (Week-long complaint, nobody pays attention)')
R('  - "Customer care call nahi uthata"')
R('    (Customer care doesn\'t answer calls)')
R('  - "Jab tak complaint nahi karenge tab tak koi nahi dekhega"')
R('    (Nobody looks until you complain)')
R()
R('  RECOMMENDED ACTION')
R('  1. 2-hour follow-up SOP: WhatsApp to customer after every complaint')
R('  2. Repeat complaint alert: 2nd ticket in 7 days = auto-escalate')
R('  3. Target: reduce AVG_TIMES_REOPENED from 0.14 to <0.05')
R('  4. Fast-track first-30-day complaints (critical retention window)')
R('  5. Track FCR at customer level (not just system-level SLA)')
R()
R('  EXPECTED IMPACT: Reducing repeat complaints by 30% could convert')
R('  200-300 Detractors to Passives, improving NPS by +3-5 points.')
R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: DRIVER #3 DEEP DIVE — NETWORK STABILITY
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  4. DRIVER DEEP-DIVE #3: NETWORK STABILITY')
R('     "Customers talk about it most, but we can\'t measure it well"')
R('=' * 76)
R()
R('  EVIDENCE STRENGTH: MEDIUM-HIGH (qual + industry agree; quant limited)')
R()
R('  THIS IS A MEASUREMENT GAP, NOT DRIVER IRRELEVANCE')
R()
R('  What customers say (STRONG signal):')
R('  - slow_speed: 877 comments (16.0% of classified) — #1 theme by volume')
R('  - disconnection_frequency: 676 comments (12.3%) — #3 theme')
R('  - internet_down_outage: 147 comments (2.7%)')
R('  - Combined network complaints: 1,700+ comments = 31% of all themes')
R('  - "100 Mbps bola tha, aaati 5-6 hi hai" (Promised 100, getting 5-6)')
R()
R('  What our models see (WEAK signal):')
R('  - avg_uptime_pct: NOT in NPS KDA consensus top-8')
R('  - Only 0.9% gap in PEAK_UPTIME_PCT between Detractors and Promoters')
R('  - BUT: avg_uptime_pct IS #3 churn predictor (11.1% importance)')
R('  - AND: For 6+ month customers, uptime becomes #1 churn predictor')
R()
R('  WHY THE GAP:')
R('  - Uptime measured at PARTNER FLEET level, not customer device level')
R('  - A partner with 71% uptime may have customers at 95% and 50%')
R('  - Speed data is COMPLETELY MISSING from Snowflake at customer level')
R('  - The #1 customer complaint has NO statistical representation')
R()
R('  INDUSTRY EXPERT VIEW')
R('  - "Connection stability > speed for this segment"')
R('  - First outage experience is Breakage Point #1')
R('  - "95% monthly uptime can hide 6 hours of prime-time downtime"')
R()
R('  RECOMMENDED ACTION')
R('  1. Deploy device-level speed testing (periodic automated tests)')
R('  2. Customer-experienced uptime tracking (device ping monitoring)')
R('  3. Peak-hour QoE monitoring (6-11 PM family usage window)')
R('  4. WiFi signal strength mapping at installation')
R()
R('  EXPECTED IMPACT: Closing this measurement gap would enable targeted')
R('  interventions for individual customers instead of partner-level averages.')
R('  Cannot quantify NPS impact until device-level data is available.')
R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: DRIVER #4 DEEP DIVE — 28-DAY BILLING
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  5. DRIVER DEEP-DIVE #4: 28-DAY BILLING CYCLE')
R('     "A self-inflicted competitive wound"')
R('=' * 76)
R()
R('  EVIDENCE STRENGTH: MEDIUM-HIGH (qual + industry strong; quant indirect)')
R()
R('  THE MATH CUSTOMERS DO')
R('  - 28-day cycle = 13 billing cycles per year')
R('  - Competitors (JioFiber, Airtel) use 30-day = 12 cycles per year')
R('  - Extra cycle cost: Rs 500-600/year at typical plan prices')
R('  - For a Rs 399/month customer, that\'s 8.3% more expensive annually')
R()
R('  QUALITATIVE EVIDENCE')
R('  - billing_28day theme: 57 comments, 73.7% Detractors, NPS 3.0')
R('  - Churn rate: 31.6% (4th highest of any theme)')
R('  - Even PROMOTERS complain about it — Kano Reverse Quality')
R('  - The irritation COMPOUNDS with tenure: 69.8% from 6+ month bucket')
R('  - Customer voice: "Ye to upbhoktaon ke saath seedha dhokha hai"')
R('    (This is outright fraud against consumers)')
R()
R('  INDUSTRY EXPERT VIEW')
R('  - Finding #1: "Strategic Vulnerability"')
R('  - "Recurring reminder that Wiom charges more than competitors"')
R('  - Every recharge is a moment where the customer does competitive math')
R('  - This is the EASIEST competitor talking point: "We charge monthly,')
R('    they charge every 28 days"')
R()
R('  WHY IT MATTERS BEYOND THE MONEY')
R('  - It signals UNFAIRNESS — customers feel cheated, not just overcharged')
R('  - It creates a negative touchpoint 13x per year')
R('  - It gives competitors an easy differentiation message')
R('  - It disproportionately hurts loyal customers who stay long enough')
R('    to notice the pattern')
R()
R('  RECOMMENDED ACTION')
R('  Option A: Move to 30-day billing cycle (clean, permanent fix)')
R('  Option B: Add 2 "bonus days" per cycle for 6+ month customers')
R('  Revenue impact: ~3.7% fewer recharge events/year (13 -> 12.2)')
R('  Offset: reduced churn + eliminated competitive disadvantage')
R()
R('  EXPECTED IMPACT: Eliminates a Reverse Quality entirely. Could prevent')
R('  18-30 churns per sprint among billing-sensitive customers.')
R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 6: DRIVER #5 DEEP DIVE — PRICE/VALUE
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  6. DRIVER DEEP-DIVE #5: PRICE/VALUE PERCEPTION')
R('     "Affordability is Wiom\'s #1 moat — protect it"')
R('=' * 76)
R()
R('  EVIDENCE STRENGTH: MEDIUM-HIGH (qual + industry strong; quant moderate)')
R()
R('  THE GOOD NEWS: AFFORDABILITY IS THE #1 PROMOTER DRIVER')
R('  - pricing_affordable theme: NPS 8.6, churn 6.4% (LOWEST of any theme)')
R('  - "Sasta hai, koi extra charge nahi" (It\'s cheap, no hidden charges)')
R('  - "Paise vasool" (Value for money)')
R('  - Wiom wins on price-to-quality ratio vs JioFiber (Rs 399 vs Rs 399+)')
R()
R('  THE BAD NEWS: PRICE SENSITIVITY IS A DOUBLE-EDGED SWORD')
R('  - pricing_expensive theme: NPS 5.3, churn 28.9%')
R('  - Customers who feel overcharged churn at 4.5x the rate of those')
R('    who feel Wiom is affordable (28.9% vs 6.4%)')
R('  - The 28-day billing cycle erodes the affordability perception')
R()
R('  INDUSTRY EXPERT VIEW')
R('  - Ranked #3 in ISP driver taxonomy (value perception)')
R('  - "Price-to-quality ratio vs alternatives is the key metric"')
R('  - Low switching costs amplify price sensitivity in this segment')
R('  - JTBD: Wiom wins on "affordable internet for children\'s education"')
R()
R('  RECOMMENDED ACTION')
R('  1. DO NOT increase prices without proportional value addition')
R('  2. Address 28-day perception (see Driver #4)')
R('  3. Transparent pricing communication at onboarding')
R('  4. Communicate total value (speed + reliability + support) not just price')
R('  5. Monitor competitor pricing in each geography monthly')
R()
R('  EXPECTED IMPACT: Protecting the affordability moat maintains Wiom\'s')
R('  primary competitive advantage. Price increases without visible value')
R('  improvements would trigger outsized churn in this price-sensitive base.')
R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 7: SEGMENTATION VIEW
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  7. CUSTOMER SEGMENTATION & RISK VIEW')
R('=' * 76)
R()

# Tenure view
R('  A. BY CUSTOMER TENURE')
R()
R('  Tenure     |   n    | Mean NPS | Churn  | Key Insight')
R('  -----------+--------+----------+--------+------------------------------------------')
if tenure_stats:
    # Tenure keys may use en-dash or hyphen; normalize for matching
    norm_map = {}
    for k in tenure_stats:
        nk = k.replace('\u2013', '-').replace('\u2014', '-')  # en-dash, em-dash -> hyphen
        norm_map[nk] = k
    order = ['1-2', '3-6', '6+']
    insights = {
        '1-2': 'Speed gap + first impression; 0-15 day intervention window',
        '3-6': 'Evaluation period; partner quality decisive',
        '6+':  'Chronic issues compound; uptime becomes #1 churn driver'
    }
    for t in order:
        real_key = norm_map.get(t, t)  # map normalized key back to original
        if real_key in tenure_stats:
            s = tenure_stats[real_key]
            ins = insights.get(t, '')
            R(f'  {t:11s}| {s["n"]:6,d} |   {s["mean_nps"]:.1f}    | {s["churn"]:.1f}%  | {ins}')
R()
R('  CRITICAL TENURE INSIGHT:')
R('  - 1-2 month customers: highest churn (35.4%). The damage happens in')
R('    the first 15 days. Speed expectation gap and partner quality decide')
R('    whether the customer stays.')
R('  - 6+ month customers: lowest churn (18.4%) but when they DO churn,')
R('    uptime is the #1 predictor — infrastructure quality matters most here.')
R('  - Driver priorities SHIFT with tenure: partner/support for new,')
R('    network reliability for loyal.')
R()

# City view
R('  B. BY CITY')
R()
R('  City       |   n    | Mean NPS | Churn  | Notes')
R('  -----------+--------+----------+--------+------------------------------------------')
if city_stats:
    for ct in sorted(city_stats.keys(), key=lambda x: city_stats[x]['n'], reverse=True):
        s = city_stats[ct]
        R(f'  {str(ct):11s}| {s["n"]:6,d} |   {s["mean_nps"]:.1f}    | {s["churn"]:.1f}%  |')
R()
R('  CITY INSIGHT: Delhi dominates (79% of respondents). Mumbai and UP are')
R('  too small for standalone driver analysis but can be used for comparison.')
R('  City is NOT a consensus KDA driver — geography matters less than')
R('  partner quality within each city.')
R()

# Risk score view
R('  C. CHURN RISK DISTRIBUTION')
R()
if risk_stats:
    R('  Risk Level |   n    |  %    ')
    R('  -----------+--------+-------')
    for r in sorted(risk_stats.keys()):
        s = risk_stats[r]
        R(f'  {str(r):11s}| {s["n"]:6,d} | {s["pct"]:.1f}%')
R()
R('  RISK INSIGHT: 25.3% of customers are Critical risk. These should be')
R('  the immediate focus of closed-loop NPS follow-up. The Low-risk 62.1%')
R('  are the stable base — protect them from erosion.')
R()

# Segments
if seg_stats:
    R('  D. CUSTOMER SEGMENTS (Data-Driven Clustering)')
    R()
    R('  Segment |   n    | Mean NPS | Churn  ')
    R('  --------+--------+----------+--------')
    for s in sorted(seg_stats.keys()):
        st = seg_stats[s]
        R(f'  {str(s):8s}| {st["n"]:6,d} |   {st["mean_nps"]:.1f}    | {st["churn"]:.1f}%')
    R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 8: INDUSTRY EXPERT INDEPENDENT VIEW
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  8. INTERNET INDUSTRY EXPERT — INDEPENDENT ASSESSMENT')
R('=' * 76)
R()
R('  The Industry Expert worked independently of both the CX and Stats teams.')
R('  Their assessment provides an external benchmark for Wiom\'s findings.')
R()
R('  WHERE WIOM\'S FINDINGS ALIGN WITH ISP BEST PRACTICE')
R('  1. Partner quality as top driver: Consistent with last-mile ISP pattern')
R('     where field technicians shape the entire service perception')
R('  2. Support effort > problem frequency: Classic finding in telecom CX')
R('  3. Affordability as #1 moat: Correct positioning for lower-middle-income')
R('  4. 28-day billing vulnerability: Known competitive risk in Indian ISP market')
R('  5. Bimodal NPS distribution: Common in emerging-market ISPs with binary')
R('     service experience (works perfectly or doesn\'t work at all)')
R()
R('  BLIND SPOTS: DRIVERS THAT SHOULD MATTER BUT WEREN\'T SURFACED')
R('  These are areas where ISP industry knowledge suggests important drivers')
R('  exist, but our data couldn\'t capture them:')
R()
R('  1. Peak-hour quality of experience (6-11 PM family usage)')
R('     - No hourly speed/uptime data at customer level')
R('     - "Slow" complaints may be peak-hour contention, not always-on issue')
R()
R('  2. DNS resolution speed')
R('     - "Pages load slow" but bandwidth may be fine')
R('     - Not measured in any Snowflake table')
R()
R('  3. WiFi interference & RF environment')
R('     - Neighbor routers, walls, microwaves affect indoor WiFi quality')
R('     - Customer blames Wiom but issue is environmental')
R()
R('  4. Device limitations')
R('     - Old smartphones can\'t utilize available bandwidth')
R('     - Customer perceives "slow internet" when device is the bottleneck')
R()
R('  5. Power supply instability')
R('     - Unstable power causes router reboots = intermittent connectivity')
R('     - Common in tier-2/3 areas (UP, parts of Delhi NCR)')
R()
R('  6. Competitive context by geography')
R('     - NPS interpretation needs local competitive benchmark')
R('     - NPS 5 where JioFiber operates is worse than NPS 5 without alternatives')
R()
R('  COMPETITIVE CONTEXT')
R('  +------------------+----------+--------+---------+-------------------+')
R('  | Competitor        | Tech     | Price  | Billing | Positioning       |')
R('  +------------------+----------+--------+---------+-------------------+')
R('  | JioFiber          | Fiber    | Rs 399+| 30-day  | Premium, reliable |')
R('  | Airtel Xstream    | Fiber    | Rs 499+| 30-day  | Premium, quality  |')
R('  | BSNL FTTH         | Fiber    | Rs 329+| 30-day  | Cheap, unreliable |')
R('  | Local Cable ISP   | Cable    | Rs 200+| Monthly | Flexible payment  |')
R('  | Wiom              | Wireless | Rs 399+| 28-day  | Affordable, fast  |')
R('  +------------------+----------+--------+---------+-------------------+')
R()
R('  Wiom\'s 28-day billing is a UNIQUE disadvantage among competitors.')
R('  Every competitor uses 30-day cycles.')
R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 9: MONITORING FRAMEWORK
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  9. MONITORING FRAMEWORK & KPI TARGETS')
R('=' * 76)
R()
R('  Recommended Metabase dashboards to track each driver:')
R()
R('  A. PARTNER QUALITY SCORECARD (Weekly)')
R('  +-----------------------------------+---------------+------------------+')
R('  | Metric                            | Current State | Target           |')
R('  +-----------------------------------+---------------+------------------+')
R('  | Partner Uptime % (30d avg)        | ~71%          | >85%             |')
R('  | Partner Install TAT (median hrs)  | TBD           | <24 hrs          |')
R('  | Partner SLA Compliance %          | TBD           | >90%             |')
R('  | Partner FCR Rate                  | TBD           | >70%             |')
R('  | Partner Customer Churn Rate       | varies        | <15%             |')
R('  | # of C-tier Partners              | TBD           | <10% of total    |')
R('  +-----------------------------------+---------------+------------------+')
R()
R('  B. SUPPORT HEALTH DASHBOARD (Daily)')
R('  +-----------------------------------+---------------+------------------+')
R('  | Metric                            | Current State | Target           |')
R('  +-----------------------------------+---------------+------------------+')
R('  | Avg Ticket Resolution TAT (hrs)   | TBD           | <4 hrs           |')
R('  | Ticket Reopen Rate                | 0.14 avg      | <0.05            |')
R('  | Repeat Complaint Rate (7-day)     | 64% of ticket | <30%             |')
R('  |   havers                          |               |                  |')
R('  | Missed Call Ratio                 | tracked       | <10%             |')
R('  | First-30-Day Complaint Rate       | TBD           | <5%              |')
R('  +-----------------------------------+---------------+------------------+')
R()
R('  C. NPS THEME TRACKER (Per Sprint)')
R('  +-----------------------------------+---------------+------------------+')
R('  | Metric                            | Current State | Target           |')
R('  +-----------------------------------+---------------+------------------+')
R('  | % Detractor (overall)             | 46.0%         | <35%             |')
R('  | % Detractor citing network issues | ~28%          | <20%             |')
R('  | % Detractor citing support issues | ~29%          | <15%             |')
R('  | 28-day billing complaint volume   | 57 (0.4%)     | 0 (after fix)    |')
R('  | Router/device complaint volume    | 26 (0.2%)     | <10              |')
R('  | Mean NPS by tenure bucket         | 6.5/5.9/5.6   | 7.0/6.5/6.5     |')
R('  +-----------------------------------+---------------+------------------+')
R()
R('  D. CHURN RISK EARLY WARNING (Weekly)')
R('  +-----------------------------------+---------------+------------------+')
R('  | Metric                            | Current State | Target           |')
R('  +-----------------------------------+---------------+------------------+')
R('  | % Critical Risk customers         | 25.3%         | <15%             |')
R('  | Days since last recharge (avg)    | monitored     | flag >10 days    |')
R('  | New customer 15-day churn rate    | ~35%          | <25%             |')
R('  | Promoter churn rate               | 20.2%         | <12%             |')
R('  +-----------------------------------+---------------+------------------+')
R()
R('  SPRINT-LEVEL KPI TARGETS (Derived from NPS Analysis)')
R()
R('  Short-term (next 3 sprints / ~6 weeks):')
R('  - Reduce repeat complaint rate by 20% (SOP implementation)')
R('  - Identify and train bottom 10% of partners')
R('  - Launch closed-loop Detractor callback program')
R('  - Target: NPS improvement of +5 points (from ~0 to +5)')
R()
R('  Medium-term (next 6 sprints / ~12 weeks):')
R('  - Partner quality tiering system operational')
R('  - 28-day billing decision made and communicated')
R('  - Device-level monitoring pilot in 2-3 high-complaint areas')
R('  - Target: NPS improvement of +10 points (to +10)')
R()
R('  Long-term (next 12 sprints / ~24 weeks):')
R('  - Full partner quality program with incentive alignment')
R('  - Proactive outage communication system live')
R('  - Device-level quality data integrated into Metabase')
R('  - Target: NPS improvement of +15-20 points (to +15-20)')
R()

# ══════════════════════════════════════════════════════════════════════════
# PAGE 10: APPENDIX — DATA QUALITY & METHODOLOGY
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  10. APPENDIX: DATA QUALITY & METHODOLOGY NOTES')
R('=' * 76)
R()
R('  A. WHAT THE DATA CAN TELL US (High Confidence)')
R('  - Which PARTNERS drive the most dissatisfaction (and churn)')
R('  - Whether SUPPORT HANDLING predicts NPS and churn')
R('  - Which TENURE STAGE is most vulnerable')
R('  - Whether 28-DAY BILLING correlates with churn')
R('  - Whether NPS SCORE adds value beyond operational data for churn')
R('  - Overall RISK SCORES for proactive intervention')
R()
R('  B. WHAT THE DATA CANNOT TELL US (Known Limitations)')
R('  1. Speed data is missing: The #1 customer complaint has no metric')
R('  2. Uptime is too coarse: Partner-level masks individual experience')
R('  3. Outage data covers only ~20 days (Dec 20 - Jan 8, 2026)')
R('  4. Only 42% of respondents wrote comments (selection bias)')
R('  5. Churn model AUC of 0.977 is high partly due to total_payments')
R('     feature (more payments = longer active = not churned)')
R('  6. Install TAT analysis has insufficient sample for new customers')
R('     (only 25 in 0-15 day bucket)')
R('  7. Competitive context is not in the data (no neighborhood-level)')
R('  8. Promoter churn (20.2%) is abnormally high and partially unexplained')
R()
R('  C. METHODOLOGY SUMMARY')
R('  - Phase 0: 13,045 NPS responses parsed from Excel, joined to Snowflake')
R('  - Phase 1: 5,488 Hindi/Hinglish/English comments processed through')
R('    NLP pipeline (cleaning, transliteration, translation, sentiment)')
R('    23 emergent themes discovered via unsupervised clustering')
R('  - Phase 2: CX frameworks applied (SERVQUAL, Kano, CES, JTBD)')
R('  - Phase 2b: Independent ISP industry expert assessment')
R('  - Phase 3: 200+ features engineered from Snowflake operational data')
R('  - Phase 4: Key Driver Analysis (4 methods: Ridge, Random Forest,')
R('    Gradient Boosting, Partial Correlation), churn models (AUC 0.977),')
R('    customer segmentation (6 clusters), temporal stability analysis')
R('  - Phase 5: Triangulation of qualitative, quantitative, and industry')
R('    expert findings; confidence-rated unified driver ranking')
R('  - Phase 6: This leadership synthesis')
R()
R('  D. DATA LEAKAGE CONTROLS')
R('  Excluded from churn model to prevent artificial accuracy:')
R('  - days_since_last_recharge (directly encodes churn definition)')
R('  - sprint_num (right-censoring proxy)')
R('  - recharge_regularity, total_recharges (activity proxies)')
R('  - Sprints 12-13 excluded (insufficient churn observation time)')
R()

# ══════════════════════════════════════════════════════════════════════════
# FINAL PAGE: THE ONE-PAGE SUMMARY
# ══════════════════════════════════════════════════════════════════════════
R('=' * 76)
R('  WIOM NPS DRIVER ANALYSIS — ONE-PAGE SUMMARY')
R('=' * 76)
R()
R(f'  NPS: {nps_metric:+.1f} | Promoters: {pct_promoter:.0f}% | Detractors: {pct_detractor:.0f}% | Churn: {churn_rate:.0f}%')
R()
R('  ROOT CAUSE: Wiom\'s NPS is dragged down by human delivery and process')
R('  failures, not primarily by network infrastructure.')
R()
R('  TOP 5 DRIVERS               ACTION                       TIMELINE')
R('  -------------------------   ---------------------------   ---------')
R('  1. Partner/Rohit Quality    Quality tiering + training    0-12 wks')
R('  2. Complaint Handling       Service recovery SOP          0-6 wks')
R('  3. Network Stability        Device-level monitoring       6-24 wks')
R('  4. 28-Day Billing Cycle     Move to 30-day billing        0-6 wks')
R('  5. Price/Value Perception   Protect affordability moat    Ongoing')
R()
R('  THE SURPRISING FINDING: NPS score adds ZERO predictive power for')
R('  churn beyond operational data. Use NPS for DIAGNOSIS (why unhappy),')
R('  not PREDICTION (who will churn). Operational dashboards are sufficient')
R('  for churn prediction.')
R()
R('  HIGHEST-ROI QUICK WIN: Implement service recovery SOP (2-hour')
R('  follow-up + repeat complaint alerts). 6.8-point NPS gap between bad')
R('  and good complaint resolution. Achievable in 2-4 weeks.')
R()
R('  BIGGEST STRUCTURAL FIX: Move to 30-day billing cycle. Eliminates a')
R('  self-inflicted competitive disadvantage that compounds with tenure')
R('  and creates 31.6% churn among affected customers.')
R()
R('  DATA GAPS TO CLOSE: Deploy device-level speed and uptime monitoring.')
R('  The two biggest customer complaints (speed, disconnections) are')
R('  invisible to our statistical models. Closing this gap would transform')
R('  Wiom\'s ability to identify and fix individual customer issues.')
R()
R('=' * 76)
R('  END OF CEO SYNTHESIS')
R('=' * 76)

save_report()
print(f"\nPhase 6 complete. Total report lines: {len(report_lines)}")
