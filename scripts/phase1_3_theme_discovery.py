"""
Phase 1.3: Adaptive Theme Discovery & Coding
=============================================
Builds an emergent theme codebook from processed NPS comments.
- Bottom-up: keyword frequency, n-gram analysis, co-occurrence clustering
- Top-down: validates against existing Tags codebook (22 tags)
- Tenure-aware: stratifies themes by customer age
- Output: coded dataset + emergent theme codebook document
"""

import sys, io, os, re, csv
from collections import Counter, defaultdict
from math import log, sqrt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

BASE = r"C:\Users\nikhi\wiom-nps-analysis"
DATA = os.path.join(BASE, "data")
OUTPUT = os.path.join(BASE, "output")

print("=" * 70)
print("PHASE 1.3: ADAPTIVE THEME DISCOVERY")
print("=" * 70)

# ── 1. Load processed comments ──────────────────────────────────────
print("\n[1/7] Loading processed comments...")
import pandas as pd
df = pd.read_csv(os.path.join(DATA, "nps_comments_processed.csv"), low_memory=False)
print(f"  Total comments: {len(df)}")

# Use translated_comment for analysis (lowercase, cleaned)
df['text_clean'] = df['translated_comment'].fillna('').astype(str).str.lower().str.strip()
df = df[df['text_clean'].str.len() > 2].copy()
print(f"  After removing empty/tiny: {len(df)}")

# ── 2. Build domain-specific stopwords + keyword lexicons ───────────
print("\n[2/7] Building keyword lexicons...")

STOPWORDS = {
    # English
    'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they',
    'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'can', 'shall', 'and', 'but', 'or', 'nor', 'not', 'no',
    'so', 'if', 'then', 'than', 'too', 'very', 'just', 'about', 'above',
    'after', 'before', 'between', 'from', 'to', 'in', 'on', 'at', 'by', 'for',
    'with', 'of', 'this', 'that', 'these', 'those', 'there', 'here', 'what',
    'which', 'who', 'whom', 'when', 'where', 'why', 'how', 'all', 'each',
    'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such', 'only',
    'also', 'as', 'up', 'out', 'over', 'under', 'again', 'further', 'once',
    'its', 'their', 'his', 'her', 'any', 'many', 'much', 'own', 'same',
    'into', 'through', 'during', 'because', 'while', 'until',
    # Hindi/Hinglish common fillers
    'hai', 'hain', 'ho', 'tha', 'thi', 'the', 'kar', 'karna', 'karte',
    'ka', 'ki', 'ke', 'ko', 'se', 'me', 'mein', 'par', 'pe', 'ne',
    'ye', 'yeh', 'wo', 'woh', 'jo', 'jab', 'tab', 'ab', 'bhi', 'aur',
    'ya', 'nahi', 'nahin', 'nhi', 'na', 'mat', 'bas', 'sirf', 'sab',
    'kuch', 'koi', 'bohot', 'bahut', 'bohut', 'bhot', 'ek', 'do', 'teen',
    'diya', 'deta', 'dete', 'liya', 'leta', 'lete', 'raha', 'rahi', 'rahe',
    'hota', 'hoti', 'hote', 'wala', 'wali', 'wale', 'isliye', 'kyunki',
    'lekin', 'magar', 'phir', 'toh', 'to', 'hi', 'he', 'mera', 'meri',
    'mere', 'apna', 'apni', 'apne', 'hamara', 'unka', 'unki', 'iska',
    'uska', 'kya', 'kaisa', 'kaise', 'kitna', 'kitni', 'kitne',
    'hum', 'main', 'mai', 'ap', 'aap', 'tum', 'log', 'logo',
    'agar', 'jaise', 'jese', 'tarah', 'jaisa', 'sabse', 'pehle',
    'baad', 'laga', 'lagta', 'lagti', 'milta', 'milti', 'milte',
    'kam', 'zyada', 'jyada', 'accha', 'acha', 'achha', 'achi', 'achhi',
    'bura', 'buri', 'kharab', 'theek', 'thik', 'ok',
    'haan', 'ji', 'sir', 'madam', 'bhai', 'sahab',
    'abhi', 'kabhi', 'please', 'thank', 'thanks', 'thankyou',
    'wiom', 'wifi', 'internet', 'net', 'service',  # domain terms handled separately
}

# Theme keyword dictionaries — initial seeds based on existing tags + domain knowledge
THEME_KEYWORDS = {
    'disconnection_frequency': {
        'primary': ['disconnect', 'disconnection', 'band', 'band ho', 'chala jata', 'ruk jata',
                     'cut', 'kat', 'kata', 'off ho', 'nahi chalta', 'nahin chalta',
                     'bar bar', 'baar baar', 'kabhi kabhi', 'roj', 'daily', 'har din',
                     'unstable', 'rukta', 'rukti', 'dropped', 'drop'],
        'secondary': ['connection', 'network', 'signal']
    },
    'slow_speed': {
        'primary': ['slow', 'speed', 'dhima', 'dheema', 'selow', 'sellow', 'buffering',
                     'buffer', 'loading', 'load nahi', 'speed kam', 'low speed',
                     'fast nahi', 'tez nahi', '2g', '3g'],
        'secondary': ['data', 'download', 'upload', 'mbps']
    },
    'good_speed': {
        'primary': ['fast', 'tez', 'speed acchi', 'speed achhi', 'speed acha',
                     'high speed', 'best speed', 'good speed', 'fast speed',
                     'speed badiya', 'speed mast', 'superb speed'],
        'secondary': ['quick', 'jaldi']
    },
    'internet_down_outage': {
        'primary': ['down', 'server down', 'outage', 'band tha', 'band hai',
                     'nahi chal raha', 'kaam nahi', 'working nahi', 'not working',
                     'chalu nahi', 'dead', 'light nahi'],
        'secondary': ['server', 'problem']
    },
    'range_coverage': {
        'primary': ['range', 'coverage', 'signal weak', 'weak signal', 'door',
                     'kamre', 'room', 'paas', 'near', 'far', 'duur',
                     'range kam', 'range nahi', 'signal nahi', 'reach'],
        'secondary': ['router', 'device', 'wall']
    },
    'complaint_resolution_bad': {
        'primary': ['complaint', 'complain', 'resolve nahi', 'solved nahi', 'fix nahi',
                     'solution nahi', 'response nahi', 'reply nahi', 'sunvai nahi',
                     'jawab nahi', 'late', 'delay', 'pending', 'no resolution',
                     'not resolved', 'still pending', 'dikkat solve nahi'],
        'secondary': ['ticket', 'issue', 'raised', 'call']
    },
    'complaint_resolution_good': {
        'primary': ['turant', 'jaldi solve', 'fast resolution', 'quickly resolved',
                     'resolved', 'fixed', 'solve ho gaya', 'theek ho gaya',
                     'toh solve', 'problem solve', 'issue resolved'],
        'secondary': ['complaint', 'quick', 'fast']
    },
    'call_center_bad': {
        'primary': ['call center', 'customer care', 'call nahi', 'phone nahi',
                     'uthata nahi', 'response nahi', 'helpline', 'support nahi',
                     'call karo', 'call karte', 'sunwai'],
        'secondary': ['number', 'agent', 'executive']
    },
    'call_center_good': {
        'primary': ['customer care accha', 'support accha', 'call pe help',
                     'responsive', 'helpful staff', 'good support',
                     'customer support acha'],
        'secondary': ['staff', 'team', 'care']
    },
    'technician_partner_bad': {
        'primary': ['technician', 'mechanic', 'rohit', 'partner', 'aadmi',
                     'aaya nahi', 'nahi aaya', 'nahi aata', 'late aaya',
                     'install wala', 'service wala', 'kaam nahi karta'],
        'secondary': ['visit', 'came', 'behavior', 'rude']
    },
    'technician_partner_good': {
        'primary': ['technician acha', 'partner acha', 'rohit acha',
                     'jaldi aaya', 'install acha', 'good installation',
                     'quick installation', 'jaldi install'],
        'secondary': ['helpful', 'polite', 'cooperative']
    },
    'pricing_expensive': {
        'primary': ['expensive', 'mehenga', 'mehnga', 'costly', 'price high',
                     'paisa jyada', 'paise zyada', 'rate high', 'charges',
                     'charge jyada', 'mahanga'],
        'secondary': ['money', 'rupee', 'rs', 'amount', 'cost']
    },
    'pricing_affordable': {
        'primary': ['affordable', 'sasta', 'cheap', 'budget', 'kam price',
                     'price acha', 'reasonable', 'value for money',
                     'kam paisa', 'free', 'no charges', 'free of cost',
                     'koi charge nahi'],
        'secondary': ['price', 'cost', 'plan']
    },
    'billing_28day': {
        'primary': ['28 day', '28 din', '28day', '28-day', '28din',
                     'mahina nahi', 'month nahi', '30 din', '30 days',
                     'monthly nahi', 'pura mahina', 'billing cycle',
                     'validity', 'plan duration', 'sirf 28'],
        'secondary': ['plan', 'recharge', 'renew', 'expire']
    },
    'rdni_recharge_no_internet': {
        'primary': ['recharge kiya', 'recharge done', 'payment done', 'paisa diya',
                     'recharge kar diya', 'recharge ke baad', 'payment ke baad',
                     'par net nahi', 'but net nahi', 'internet nahi', 'nahi chala',
                     'activate nahi', 'plan activate'],
        'secondary': ['recharge', 'payment', 'money']
    },
    'general_positive': {
        'primary': ['good', 'best', 'excellent', 'great', 'awesome', 'nice',
                     'badiya', 'badhiya', 'mast', 'shandar', 'jabardast',
                     'acchi service', 'acha service', 'best service',
                     'satisfied', 'happy', 'khush', 'pasand'],
        'secondary': ['recommend', 'suggest', 'like']
    },
    'general_negative': {
        'primary': ['bad', 'worst', 'terrible', 'horrible', 'pathetic',
                     'bekar', 'bakwas', 'ghatiya', 'bekaar',
                     'zero', 'fraud', 'dhokha', 'loot', 'scam',
                     'band karo', 'hatao', 'chhodo'],
        'secondary': ['never', 'waste', 'useless']
    },
    'app_tech_issues': {
        'primary': ['app', 'application', 'software', 'update', 'login',
                     'recharge option', 'app nahi', 'app kharab', 'crash',
                     'hang', 'bug', 'error'],
        'secondary': ['mobile', 'phone', 'online']
    },
    'ott_content': {
        'primary': ['ott', 'tv', 'movie', 'hotstar', 'netflix', 'youtube',
                     'streaming', 'channel', 'content', 'entertainment'],
        'secondary': ['watch', 'video', 'show']
    },
    'shifting_relocation': {
        'primary': ['shifting', 'shift', 'relocate', 'ghar badal', 'move',
                     'new address', 'new location', 'transfer'],
        'secondary': ['address', 'location', 'area']
    },
    'router_device': {
        'primary': ['router', 'netbox', 'device', 'box', 'modem',
                     'light', 'blink', 'restart', 'reset', 'reboot',
                     'power', 'cable', 'wire', 'adapter'],
        'secondary': ['hardware', 'equipment']
    },
    'autopay_payment_mode': {
        'primary': ['autopay', 'auto pay', 'online payment', 'cash',
                     'upi', 'paytm', 'phonepe', 'gpay', 'google pay',
                     'payment method', 'payment mode', 'recharge method'],
        'secondary': ['pay', 'payment']
    },
    'competitor_comparison': {
        'primary': ['jio', 'jiofiber', 'airtel', 'bsnl', 'hathway', 'act',
                     'other company', 'dusri company', 'competitor',
                     'better option', 'switch', 'change provider'],
        'secondary': ['compare', 'comparison', 'alternative']
    },
}

# ── 3. Score each comment against all themes ────────────────────────
print("\n[3/7] Scoring comments against theme lexicons...")

def score_theme(text, theme_kws):
    """Score a text against a theme's keyword dictionary."""
    score = 0
    matched_keywords = []
    text_lower = text.lower()

    for kw in theme_kws.get('primary', []):
        if kw in text_lower:
            score += 2
            matched_keywords.append(kw)

    for kw in theme_kws.get('secondary', []):
        if kw in text_lower:
            score += 1
            matched_keywords.append(kw)

    return score, matched_keywords

# Score every comment against every theme
theme_scores = {}
for theme_name, theme_kws in THEME_KEYWORDS.items():
    scores = []
    for idx, row in df.iterrows():
        text = str(row['text_clean'])
        s, kws = score_theme(text, theme_kws)
        scores.append(s)
    theme_scores[theme_name] = scores

# Add theme scores to dataframe
for theme_name, scores in theme_scores.items():
    df[f'theme_score_{theme_name}'] = scores

# Assign primary theme (highest score > 0)
score_cols = [c for c in df.columns if c.startswith('theme_score_')]
df['primary_theme'] = 'unclassified'
df['primary_theme_score'] = 0
df['secondary_theme'] = ''
df['secondary_theme_score'] = 0

for idx in df.index:
    scores = [(col.replace('theme_score_', ''), df.at[idx, col]) for col in score_cols]
    scores.sort(key=lambda x: x[1], reverse=True)

    if scores[0][1] > 0:
        df.at[idx, 'primary_theme'] = scores[0][0]
        df.at[idx, 'primary_theme_score'] = scores[0][1]

    if len(scores) > 1 and scores[1][1] > 0:
        df.at[idx, 'secondary_theme'] = scores[1][0]
        df.at[idx, 'secondary_theme_score'] = scores[1][1]

# ── 4. Analyze theme distribution ──────────────────────────────────
print("\n[4/7] Analyzing theme distributions...")

theme_dist = df['primary_theme'].value_counts()
print(f"\n  Primary Theme Distribution:")
for theme, count in theme_dist.items():
    pct = count / len(df) * 100
    print(f"    {theme:40s}: {count:5d} ({pct:5.1f}%)")

# ── 5. Theme x NPS Group cross-tab ──────────────────────────────────
print("\n[5/7] Building theme cross-tabs...")

# Theme x NPS Group
theme_nps = pd.crosstab(df['primary_theme'], df['nps_group'], normalize='index') * 100
theme_nps_counts = pd.crosstab(df['primary_theme'], df['nps_group'])

# Theme x Tenure
theme_tenure = pd.crosstab(df['primary_theme'], df['tenure_excel'], normalize='index') * 100

# Theme x Churn
theme_churn = df.groupby('primary_theme').agg(
    total=('is_churned', 'count'),
    churned=('is_churned', 'sum'),
    mean_nps=('nps_score', 'mean')
).reset_index()
theme_churn['churn_rate'] = (theme_churn['churned'] / theme_churn['total'] * 100).round(1)
theme_churn = theme_churn.sort_values('churn_rate', ascending=False)

# Theme x Sentiment
theme_sentiment = pd.crosstab(df['primary_theme'], df['sentiment_polarity'], normalize='index') * 100

# Theme x City
theme_city = pd.crosstab(df['primary_theme'], df['city'], normalize='index') * 100

# ── 6. Discover emergent sub-themes via word frequency within themes ─
print("\n[6/7] Discovering sub-themes via word frequency analysis...")

# For each major theme, find the most common words (beyond keywords)
theme_top_words = {}
for theme in df['primary_theme'].unique():
    if theme == 'unclassified':
        continue
    theme_texts = df[df['primary_theme'] == theme]['text_clean'].tolist()
    word_counter = Counter()
    for text in theme_texts:
        words = re.findall(r'[a-zA-Z\u0900-\u097F]+', text.lower())
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        word_counter.update(words)
    theme_top_words[theme] = word_counter.most_common(20)

# Analyze unclassified comments for missing themes
unclassified = df[df['primary_theme'] == 'unclassified']
print(f"\n  Unclassified comments: {len(unclassified)} ({len(unclassified)/len(df)*100:.1f}%)")

uncl_words = Counter()
for text in unclassified['text_clean'].tolist():
    words = re.findall(r'[a-zA-Z\u0900-\u097F]+', text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    uncl_words.update(words)

print(f"  Top words in unclassified comments:")
for word, count in uncl_words.most_common(30):
    print(f"    {word:25s}: {count}")

# ── 7. Build representative examples for each theme ─────────────────
print("\n[7/7] Selecting representative examples...")

# For each theme, pick 3 high-quality examples
theme_examples = {}
for theme in df['primary_theme'].unique():
    if theme == 'unclassified':
        continue
    theme_df = df[df['primary_theme'] == theme].sort_values('primary_theme_score', ascending=False)
    # Get high-quality first, then medium
    hq = theme_df[theme_df['comment_quality'] == 'high'].head(2)
    mq = theme_df[theme_df['comment_quality'] == 'medium'].head(2)
    examples = pd.concat([hq, mq]).head(3)
    theme_examples[theme] = []
    for _, row in examples.iterrows():
        theme_examples[theme].append({
            'nps_score': row['nps_score'],
            'nps_group': row['nps_group'],
            'original': str(row['oe_raw'])[:200],
            'translated': str(row['translated_comment'])[:200],
            'language': row['detected_language'],
            'sentiment': row['sentiment_polarity'],
            'emotion': row['emotion'],
            'city': row['city'],
            'tenure': row['tenure_excel']
        })

# ── Save coded dataset ──────────────────────────────────────────────
print("\n  Saving coded dataset...")
output_cols = ['phone_number', 'nps_score', 'nps_group', 'oe_raw', 'translated_comment',
               'detected_language', 'sentiment_polarity', 'sentiment_intensity', 'emotion',
               'comment_quality', 'score_sentiment_mismatch',
               'primary_theme', 'primary_theme_score', 'secondary_theme', 'secondary_theme_score',
               'mentions_28day', 'mentions_competitor', 'competitor_names', 'mentions_amount',
               'tenure_bucket', 'tenure_excel', 'tenure_days', 'city', 'churn_label', 'is_churned',
               'Sprint ID', 'sprint_num', 'Channel', 'first_time_wifi',
               'NPS Reason - Primary', 'Primary Category']

existing_cols = [c for c in output_cols if c in df.columns]
df[existing_cols].to_csv(os.path.join(DATA, "nps_comments_themed.csv"), index=False, encoding='utf-8-sig')
print(f"  Saved nps_comments_themed.csv ({len(df)} rows)")

# ── Generate comprehensive report ───────────────────────────────────
report_lines = []
r = report_lines.append

r("=" * 70)
r("PHASE 1.3 — ADAPTIVE THEME DISCOVERY REPORT")
r("=" * 70)
r("")
r(f"Total comments analyzed: {len(df)}")
r(f"Classified: {len(df[df['primary_theme'] != 'unclassified'])} ({len(df[df['primary_theme'] != 'unclassified'])/len(df)*100:.1f}%)")
r(f"Unclassified: {len(unclassified)} ({len(unclassified)/len(df)*100:.1f}%)")
r(f"With secondary theme: {len(df[df['secondary_theme'] != ''])} ({len(df[df['secondary_theme'] != ''])/len(df)*100:.1f}%)")
r("")

# ── Theme Distribution ──
r("=" * 70)
r("THEME DISTRIBUTION (Primary Theme)")
r("=" * 70)
for theme, count in theme_dist.items():
    pct = count / len(df) * 100
    bar = "#" * int(pct)
    r(f"  {theme:40s}: {count:5d} ({pct:5.1f}%) {bar}")

# ── Theme x NPS Group ──
r("")
r("=" * 70)
r("THEME x NPS GROUP (% within each theme)")
r("=" * 70)
r(f"{'Theme':40s} | {'Promoter':>10s} | {'Passive':>10s} | {'Detractor':>10s} | {'Count':>6s} | {'Mean NPS':>8s}")
r("-" * 100)
for theme in theme_dist.index:
    if theme in theme_nps.index:
        prom = theme_nps.at[theme, 'Promoter'] if 'Promoter' in theme_nps.columns else 0
        pas = theme_nps.at[theme, 'Passive'] if 'Passive' in theme_nps.columns else 0
        det = theme_nps.at[theme, 'Detractor'] if 'Detractor' in theme_nps.columns else 0
        cnt = theme_nps_counts.loc[theme].sum()
        mn = df[df['primary_theme'] == theme]['nps_score'].mean()
        r(f"  {theme:40s} | {prom:9.1f}% | {pas:9.1f}% | {det:9.1f}% | {cnt:6d} | {mn:8.1f}")

# ── Theme x Churn ──
r("")
r("=" * 70)
r("THEME x CHURN RATE (sorted by churn rate)")
r("=" * 70)
r(f"{'Theme':40s} | {'Total':>6s} | {'Churned':>7s} | {'Churn%':>7s} | {'Mean NPS':>8s}")
r("-" * 80)
for _, row in theme_churn.iterrows():
    r(f"  {row['primary_theme']:40s} | {int(row['total']):6d} | {int(row['churned']):7d} | {row['churn_rate']:6.1f}% | {row['mean_nps']:8.1f}")

# ── Theme x Tenure ──
r("")
r("=" * 70)
r("THEME x TENURE BUCKET (% within each theme)")
r("=" * 70)
tenure_cols_available = [c for c in ['1\u20132', '3\u20136', '6+'] if c in theme_tenure.columns]
header = f"{'Theme':40s}"
for tc in tenure_cols_available:
    header += f" | {tc:>8s}"
r(header)
r("-" * (42 + len(tenure_cols_available) * 12))
for theme in theme_dist.index:
    if theme in theme_tenure.index:
        line = f"  {theme:40s}"
        for tc in tenure_cols_available:
            val = theme_tenure.at[theme, tc] if tc in theme_tenure.columns else 0
            line += f" | {val:7.1f}%"
        r(line)

# ── Theme x City ──
r("")
r("=" * 70)
r("THEME x CITY (% within each theme)")
r("=" * 70)
city_cols_available = [c for c in ['Delhi', 'Mumbai', 'UP'] if c in theme_city.columns]
header = f"{'Theme':40s}"
for cc in city_cols_available:
    header += f" | {cc:>8s}"
r(header)
r("-" * (42 + len(city_cols_available) * 12))
for theme in theme_dist.index:
    if theme in theme_city.index:
        line = f"  {theme:40s}"
        for cc in city_cols_available:
            val = theme_city.at[theme, cc] if cc in theme_city.columns else 0
            line += f" | {val:7.1f}%"
        r(line)

# ── Theme x Sentiment ──
r("")
r("=" * 70)
r("THEME x SENTIMENT (% within each theme)")
r("=" * 70)
r(f"{'Theme':40s} | {'Positive':>10s} | {'Neutral':>10s} | {'Negative':>10s}")
r("-" * 80)
for theme in theme_dist.index:
    if theme in theme_sentiment.index:
        pos = theme_sentiment.at[theme, 'positive'] if 'positive' in theme_sentiment.columns else 0
        neu = theme_sentiment.at[theme, 'neutral'] if 'neutral' in theme_sentiment.columns else 0
        neg = theme_sentiment.at[theme, 'negative'] if 'negative' in theme_sentiment.columns else 0
        r(f"  {theme:40s} | {pos:9.1f}% | {neu:9.1f}% | {neg:9.1f}%")

# ── DETRACTOR DEEP DIVE: Top themes for Detractors ──
r("")
r("=" * 70)
r("DETRACTOR DEEP DIVE — Top themes among Detractors")
r("=" * 70)
det_themes = df[df['nps_group'] == 'Detractor']['primary_theme'].value_counts()
for theme, count in det_themes.items():
    pct = count / len(df[df['nps_group'] == 'Detractor']) * 100
    churn_sub = df[(df['primary_theme'] == theme) & (df['nps_group'] == 'Detractor')]
    cr = churn_sub['is_churned'].mean() * 100
    r(f"  {theme:40s}: {count:5d} ({pct:5.1f}%) | Churn: {cr:.1f}%")

# ── PROMOTER DEEP DIVE: Top themes for Promoters ──
r("")
r("=" * 70)
r("PROMOTER DEEP DIVE — Top themes among Promoters")
r("=" * 70)
prom_themes = df[df['nps_group'] == 'Promoter']['primary_theme'].value_counts()
for theme, count in prom_themes.items():
    pct = count / len(df[df['nps_group'] == 'Promoter']) * 100
    churn_sub = df[(df['primary_theme'] == theme) & (df['nps_group'] == 'Promoter')]
    cr = churn_sub['is_churned'].mean() * 100
    r(f"  {theme:40s}: {count:5d} ({pct:5.1f}%) | Churn: {cr:.1f}%")

# ── THEME EXAMPLES ──
r("")
r("=" * 70)
r("REPRESENTATIVE EXAMPLES BY THEME")
r("=" * 70)
for theme, examples in sorted(theme_examples.items()):
    r(f"\n--- {theme.upper()} ---")
    for i, ex in enumerate(examples, 1):
        r(f"  [{i}] NPS={ex['nps_score']} | {ex['nps_group']} | {ex['language']} | {ex['sentiment']} | {ex['emotion']} | {ex['city']} | Tenure: {ex['tenure']}")
        r(f"      Original: \"{ex['original']}\"")
        r(f"      Translated: \"{ex['translated']}\"")
        r("")

# ── EMERGENT THEME CODEBOOK ──
r("")
r("=" * 70)
r("EMERGENT THEME CODEBOOK")
r("=" * 70)
r("")
r("NEGATIVE EXPERIENCE THEMES:")
r("-" * 40)

neg_themes = [
    ('disconnection_frequency', 'Bad Internet Experience',
     'Frequent internet disconnections, unstable connection, network dropping repeatedly'),
    ('slow_speed', 'Bad Internet Experience',
     'Internet speed is slow, buffering, loading issues, speed below expectations'),
    ('internet_down_outage', 'Bad Internet Experience',
     'Internet completely not working, server down, extended outage periods'),
    ('range_coverage', 'Bad Internet Experience',
     'WiFi signal weak, does not reach other rooms, limited range from router'),
    ('complaint_resolution_bad', 'Bad CX Support',
     'Complaints not resolved, no response after raising issue, pending resolution, delayed fixes'),
    ('call_center_bad', 'Bad CX Support',
     'Call center not picking up, customer care not responsive, support staff unhelpful'),
    ('technician_partner_bad', 'Bad CX Support',
     'Field technician not visiting, late arrival, poor quality work, rude behavior'),
    ('pricing_expensive', 'Offering/Plan Dissatisfaction',
     'Service considered expensive, high charges, price not justified for quality'),
    ('billing_28day', 'Offering/Plan Dissatisfaction',
     '28-day billing cycle instead of calendar month, customers feel cheated paying for 13 months/year'),
    ('rdni_recharge_no_internet', 'Bad Service',
     'Customer recharged/paid but internet did not activate or still not working'),
    ('general_negative', 'Bad Internet Experience',
     'General negative sentiment about service without specifying particular issue'),
    ('app_tech_issues', 'Application/Tech',
     'App not working, recharge option missing, app crashes or errors'),
    ('shifting_relocation', 'Offering/Plan Dissatisfaction',
     'Connection transfer not feasible when customer relocates'),
    ('router_device', 'Bad Internet Experience',
     'Issues with router hardware — lights, power, restarting, cable problems'),
]

for theme_id, category, description in neg_themes:
    count = len(df[df['primary_theme'] == theme_id])
    if count > 0:
        pct = count / len(df) * 100
        r(f"\n  Theme: {theme_id}")
        r(f"  Category: {category}")
        r(f"  Count: {count} ({pct:.1f}%)")
        r(f"  Description: {description}")
        # Top words
        if theme_id in theme_top_words:
            words = ', '.join([f"{w}({c})" for w, c in theme_top_words[theme_id][:10]])
            r(f"  Top words: {words}")

r("")
r("POSITIVE EXPERIENCE THEMES:")
r("-" * 40)

pos_themes = [
    ('general_positive', 'Good Internet Experience',
     'General positive sentiment — good service, satisfied, happy with Wiom'),
    ('good_speed', 'Good Internet Experience',
     'Explicit praise for fast internet speed'),
    ('pricing_affordable', 'Good Price',
     'Appreciation of affordable pricing, value for money, low cost'),
    ('complaint_resolution_good', 'Good CX Support',
     'Complaints resolved quickly, fast issue resolution'),
    ('call_center_good', 'Good CX Support',
     'Good customer support experience, helpful call center'),
    ('technician_partner_good', 'Good CX Support',
     'Positive experience with field technician, good installation, quick visit'),
]

for theme_id, category, description in pos_themes:
    count = len(df[df['primary_theme'] == theme_id])
    if count > 0:
        pct = count / len(df) * 100
        r(f"\n  Theme: {theme_id}")
        r(f"  Category: {category}")
        r(f"  Count: {count} ({pct:.1f}%)")
        r(f"  Description: {description}")
        if theme_id in theme_top_words:
            words = ', '.join([f"{w}({c})" for w, c in theme_top_words[theme_id][:10]])
            r(f"  Top words: {words}")

r("")
r("OTHER THEMES:")
r("-" * 40)

other_themes = [
    ('ott_content', 'Feature Request', 'OTT/streaming content requests or complaints'),
    ('autopay_payment_mode', 'Payment', 'Comments about payment methods — autopay, UPI, cash'),
    ('competitor_comparison', 'Competitive', 'Comparisons with JioFiber, Airtel, or other providers'),
]

for theme_id, category, description in other_themes:
    count = len(df[df['primary_theme'] == theme_id])
    if count > 0:
        pct = count / len(df) * 100
        r(f"\n  Theme: {theme_id}")
        r(f"  Category: {category}")
        r(f"  Count: {count} ({pct:.1f}%)")
        r(f"  Description: {description}")

# ── UNCLASSIFIED ANALYSIS ──
r("")
r("=" * 70)
r("UNCLASSIFIED COMMENT ANALYSIS")
r("=" * 70)
r(f"Total unclassified: {len(unclassified)} ({len(unclassified)/len(df)*100:.1f}%)")
r("")
r("NPS Group distribution of unclassified:")
uncl_nps = unclassified['nps_group'].value_counts()
for grp, cnt in uncl_nps.items():
    r(f"  {grp}: {cnt} ({cnt/len(unclassified)*100:.1f}%)")

r("")
r("Top 30 words in unclassified comments:")
for word, count in uncl_words.most_common(30):
    r(f"  {word:25s}: {count}")

r("")
r("Sample unclassified comments (first 20):")
for i, (_, row) in enumerate(unclassified.head(20).iterrows()):
    r(f"  [{i+1}] NPS={row['nps_score']} | {row['nps_group']} | \"{str(row['oe_raw'])[:120]}\"")

# ── CROSS-VALIDATION WITH EXISTING TAGS ──
r("")
r("=" * 70)
r("CROSS-VALIDATION: Our Themes vs Existing Pre-Coded Categories")
r("=" * 70)
r("")
r("Mapping between our emergent themes and Wiom's existing 'NPS Reason - Primary' codes:")
r("")

# For rows that have both our theme and existing codes
coded = df[df['NPS Reason - Primary'].notna() & (df['primary_theme'] != 'unclassified')]
if len(coded) > 0:
    cross = pd.crosstab(coded['primary_theme'], coded['NPS Reason - Primary'])
    r(f"Rows with both theme and pre-code: {len(coded)}")
    r("")
    # For each of our themes, show top matching pre-codes
    for theme in cross.index:
        top_codes = cross.loc[theme].sort_values(ascending=False)
        top_codes = top_codes[top_codes > 0].head(5)
        r(f"  {theme}:")
        for code, cnt in top_codes.items():
            r(f"    -> {code}: {cnt}")
        r("")

# ── KEY FINDINGS SUMMARY ──
r("")
r("=" * 70)
r("KEY FINDINGS SUMMARY")
r("=" * 70)
r("")

# Find highest-churn themes
r("1. HIGHEST CHURN THEMES (actionable intervention targets):")
high_churn = theme_churn[theme_churn['total'] >= 20].sort_values('churn_rate', ascending=False).head(10)
for _, row in high_churn.iterrows():
    r(f"   {row['primary_theme']:40s}: {row['churn_rate']:.1f}% churn (n={int(row['total'])})")

r("")
r("2. MOST COMMON DETRACTOR THEMES (volume-based priority):")
for theme, count in det_themes.head(10).items():
    pct = count / len(df[df['nps_group'] == 'Detractor']) * 100
    r(f"   {theme:40s}: {count:5d} ({pct:.1f}% of Detractors)")

r("")
r("3. THEME OVERLAP (multi-issue customers):")
multi = len(df[df['secondary_theme'] != ''])
r(f"   {multi} comments ({multi/len(df)*100:.1f}%) mention 2+ themes")
r(f"   Most common primary+secondary combinations:")
combos = df[df['secondary_theme'] != ''].groupby(['primary_theme', 'secondary_theme']).size().sort_values(ascending=False).head(15)
for (p, s), count in combos.items():
    r(f"   {p} + {s}: {count}")

r("")
r("4. SENTIMENT VALIDATION:")
r("   Themes where sentiment aligns with expectation (high confidence):")
for theme in ['disconnection_frequency', 'slow_speed', 'general_positive', 'good_speed', 'pricing_affordable']:
    if theme in theme_sentiment.index:
        neg = theme_sentiment.at[theme, 'negative'] if 'negative' in theme_sentiment.columns else 0
        pos = theme_sentiment.at[theme, 'positive'] if 'positive' in theme_sentiment.columns else 0
        r(f"   {theme:40s}: {neg:.0f}% negative, {pos:.0f}% positive")

r("")
r("5. THEMES BY CITY (differential priorities):")
for city in ['Delhi', 'Mumbai', 'UP']:
    city_df = df[df['city'] == city]
    if len(city_df) > 0:
        city_det = city_df[city_df['nps_group'] == 'Detractor']['primary_theme'].value_counts().head(5)
        r(f"\n   {city} — Top Detractor themes:")
        for theme, count in city_det.items():
            r(f"     {theme:40s}: {count}")

# Save report
report_path = os.path.join(OUTPUT, "phase1_3_theme_discovery.txt")
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))

print(f"\n  Report saved to {report_path}")

# Print summary to stdout
for line in report_lines[:60]:
    print(line)

print("\n... [full report saved to file] ...")
print("\n" + "=" * 70)
print("PHASE 1.3 COMPLETE")
print("=" * 70)
