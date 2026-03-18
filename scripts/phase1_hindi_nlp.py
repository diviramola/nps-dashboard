"""
Phase 1: Hindi/Hinglish NPS Comment Processing
===============================================
Wiom NPS Driver Analysis
Owner: Hindi Language Expert

Processes all OE comments: language detection, translation,
sentiment analysis, entity extraction, comment quality scoring.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import re
import sys
import io
import os
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Paths
DATA_DIR = r'C:\Users\nikhi\wiom-nps-analysis\data'
OUTPUT_DIR = r'C:\Users\nikhi\wiom-nps-analysis\output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("PHASE 1: Hindi/Hinglish NPS Comment Processing")
print("=" * 70)

# ──────────────────────────────────────────────────────────────────────
# 1. Load clean NPS base (from Phase 0)
# ──────────────────────────────────────────────────────────────────────
print("\n[1/8] Loading clean NPS base...")
df_all = pd.read_csv(os.path.join(DATA_DIR, 'nps_clean_base.csv'), encoding='utf-8-sig')
print(f"  Total rows: {len(df_all)}")

# Filter to rows with OE comments
df = df_all[df_all['has_comment'] == True].copy()
print(f"  Rows with comments: {len(df)}")

# Clean OE text
df['oe_raw'] = df['OE'].astype(str).str.strip()
df = df[df['oe_raw'].str.len() > 0].copy()
print(f"  After removing empty: {len(df)}")

# ──────────────────────────────────────────────────────────────────────
# 2. Language Detection
# ──────────────────────────────────────────────────────────────────────
print("\n[2/8] Detecting languages...")

# Devanagari Unicode range
DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')
# Common Hindi/Hinglish words (romanized)
HINDI_WORDS = set([
    'hai', 'nahi', 'nhi', 'bahut', 'bht', 'bohot', 'accha', 'acha', 'achha',
    'theek', 'thik', 'mera', 'hamara', 'kya', 'aur', 'lekin', 'iska', 'uska',
    'ye', 'yeh', 'ho', 'kar', 'raha', 'rahi', 'wala', 'wali', 'jata', 'jati',
    'chalta', 'chal', 'dikkat', 'paisa', 'paise', 'mahina', 'din', 'sala',
    'kharab', 'bekar', 'sahi', 'badhiya', 'badiya', 'zyada', 'kam', 'bilkul',
    'ekdum', 'kabhi', 'hota', 'hoti', 'koi', 'abhi', 'jab', 'tab', 'toh',
    'bhi', 'se', 'ka', 'ki', 'ke', 'ko', 'me', 'mai', 'mein', 'par', 'pe',
    'woh', 'wo', 'yaha', 'waha', 'sab', 'kuch', 'kuchh', 'ek', 'do', 'teen',
    'rehta', 'rehti', 'deta', 'deti', 'leta', 'leti', 'jaise', 'kaisa', 'kaisi',
    'kaese', 'kaise', 'lagta', 'lagti', 'milta', 'milti', 'baar', 'pehle',
    'baad', 'wapas', 'jayega', 'jaayega', 'chahiye', 'lagwaya', 'karwaya',
    'krte', 'krta', 'hain', 'tha', 'thi', 'the', 'nahin', 'nai', 'mat',
    'banda', 'log', 'logo', 'apna', 'apni', 'apne', 'hum', 'tum', 'main',
    'unka', 'unki', 'inhe', 'iska', 'jaise', 'waisa', 'aise', 'kitna', 'itna',
    'bohut', 'bahot', 'bhut', 'boht', 'bada', 'chhota', 'lamba', 'sasta',
    'mehnga', 'mahanga', 'sasta', 'dheere', 'jaldi', 'roz', 'ruk', 'band',
    'chalu', 'lagao', 'lagaya', 'hatao', 'diya', 'liya', 'gaya', 'aaya',
    'jaata', 'aata', 'sunna', 'suno', 'bolo', 'batao', 'dekho', 'chalo',
    'kyunki', 'isliye', 'warna', 'phir', 'fir', 'dobara', 'kabhi', 'hamesha',
    'baar', 'ek', 'agar', 'toh', 'waise', 'sirf', 'bas', 'matlab', 'samajh',
    'pata', 'pta', 'aapka', 'aap', 'tumhara', 'karo', 'kijiye', 'dijiye',
    'bataiye', 'suniye', 'dekhiye', 'kariye', 'rakho', 'bhejo', 'lo', 'do',
    'dunga', 'karunga', 'jaaunga', 'chahta', 'chahti', 'pasand', 'napasand',
    'internet', 'net', 'wifi', 'recharge', 'plan', 'speed', 'slow', 'fast',
    'connection', 'network', 'signal', 'complaint', 'customer', 'service',
])

def detect_language(text):
    if not isinstance(text, str) or len(text.strip()) == 0:
        return 'unknown'

    has_devanagari = bool(DEVANAGARI_RE.search(text))
    words = re.findall(r'[a-zA-Z]+', text.lower())
    hindi_count = sum(1 for w in words if w in HINDI_WORDS)
    total_latin = len(words)

    if has_devanagari and total_latin > 3:
        return 'mixed'
    elif has_devanagari:
        return 'hindi_devanagari'
    elif total_latin > 0 and hindi_count / max(total_latin, 1) > 0.25:
        return 'hinglish'
    elif total_latin > 0:
        return 'english'
    else:
        return 'unknown'

df['detected_language'] = df['oe_raw'].apply(detect_language)
print(f"  Language distribution:")
print(df['detected_language'].value_counts().to_string())

# ──────────────────────────────────────────────────────────────────────
# 3. Translation Dictionary
# ──────────────────────────────────────────────────────────────────────
print("\n[3/8] Building translation dictionary...")

TRANSLATION_DICT = {
    # Network quality
    'net nahi chal raha': 'internet not working',
    'net nhi chal rha': 'internet not working',
    'net nahi chalta': 'internet does not work',
    'net nhi chalta': 'internet does not work',
    'wifi nahi chalta': 'wifi does not work',
    'wifi nhi chalta': 'wifi does not work',
    'internet nahi chalta': 'internet does not work',
    'speed bahut slow': 'speed very slow',
    'speed bahut kam': 'speed very low',
    'speed kam hai': 'speed is low',
    'speed slow hai': 'speed is slow',
    'speed achi hai': 'speed is good',
    'speed acchi hai': 'speed is good',
    'speed sahi hai': 'speed is fine',
    'band ho jata hai': 'stops working/disconnects',
    'band ho jaata hai': 'stops working/disconnects',
    'band ho jata': 'stops working',
    'band ho jaata': 'stops working',
    'chalte chalte band': 'stops while running',
    'chalte chalte band ho jata': 'disconnects while running',
    'range nahi aati': 'no range/signal',
    'range nhi aati': 'no range/signal',
    'range issue': 'range/signal problem',
    'network issue': 'network problem',
    'network problem': 'network problem',
    'network shi hai': 'network is fine',
    'network sahi hai': 'network is fine',
    'network theek hai': 'network is okay',
    'internet down': 'internet is down',
    'net down': 'internet is down',
    'disconnect hota hai': 'gets disconnected',
    'disconnect ho jata': 'gets disconnected',
    'buffering hota hai': 'buffering happens',
    'buffering aata hai': 'buffering occurs',
    'chal nhi raha': 'not working',
    'chal nahi raha': 'not working',
    'nahi chal raha': 'not working',
    'nhi chal rha': 'not working',
    'chalta hi nahi': 'does not work at all',
    'bilkul nahi chalta': 'does not work at all',
    'badhiya chalta hai': 'works great',
    'accha chalta hai': 'works well',
    'acha chalta hai': 'works well',
    'sahi chalta hai': 'works fine',
    'theek chalta hai': 'works okay',
    'smooth chalta hai': 'runs smoothly',
    'smooth chalta': 'runs smooth',
    'ganda chalta hai': 'works poorly',
    'ganda chalta': 'works poorly',

    # Pricing
    'recharge mehnga hai': 'recharge is expensive',
    'recharge mehnga': 'recharge expensive',
    'mehnga hai': 'is expensive',
    'mahanga hai': 'is expensive',
    'sasta hai': 'is affordable/cheap',
    'paisa vasool': 'value for money',
    'paisa vasool hai': 'is value for money',
    'costly hai': 'is costly',
    'affordable hai': 'is affordable',
    'rate badha diya': 'rate has been increased',
    'rate zyada hai': 'rate is too high',
    'paise zyada lagte': 'costs too much',

    # Plans
    '28 din ka plan': '28-day plan',
    '28 din ka': '28-day',
    'mahina pura nahi': 'month not complete',
    'mahina pura nhi': 'month not complete',
    'plan chhota hai': 'plan is small/short',
    'data khatam': 'data finished/exhausted',
    'data khatam ho jata': 'data runs out',
    'data khatam ho jaata': 'data runs out',
    'unlimited data': 'unlimited data',
    'plan accha hai': 'plan is good',
    'plan sahi hai': 'plan is fine',
    'plan theek hai': 'plan is okay',

    # Support / Complaints
    'complaint ki': 'filed a complaint',
    'complaint ki thi': 'had filed a complaint',
    'complaint kiya': 'filed a complaint',
    'complaint kar diya': 'filed a complaint',
    'koi sunata nahi': 'nobody listens',
    'koi sunta nahi': 'nobody listens',
    'baat nahi sun rahe': 'they do not listen',
    'baat nhi sunte': 'they do not listen',
    'call nahi uthate': 'do not pick up calls',
    'call nhi uthate': 'do not pick up calls',
    'solution nahi mila': 'no solution received',
    'solution nhi mila': 'no solution received',
    'paisa wapas nahi': 'money not returned',
    'paisa wapas nahi mila': 'did not get money back',
    'refund nahi mila': 'refund not received',
    'customer support': 'customer support',
    'customer care': 'customer care',

    # Installation
    'aadmi nahi aaya': 'technician did not come',
    'aadmi nhi aaya': 'technician did not come',
    'install mein dikkat': 'problem in installation',
    'lagwaya tha': 'had it installed',
    'connection lagwaya': 'got connection installed',
    'naya connection': 'new connection',

    # General positive
    'badhiya hai': 'is great',
    'badhiya chalta': 'works great',
    'accha hai': 'is good',
    'acha hai': 'is good',
    'theek hai': 'is okay',
    'thik hai': 'is okay',
    'sab sahi hai': 'everything is fine',
    'sab theek hai': 'everything is okay',
    'best hai': 'is the best',
    'bahut accha': 'very good',
    'bahut acha': 'very good',
    'mast hai': 'is awesome',
    'maza aa gaya': 'had a great time',
    'khush hai': 'am happy',
    'khush hoon': 'am happy',
    'satisfied': 'satisfied',
    'koi dikkat nahi': 'no problems',
    'koi problem nahi': 'no problems',

    # General negative
    'kharab hai': 'is bad',
    'bekar hai': 'is useless',
    'bakwas hai': 'is rubbish',
    'bakwas': 'rubbish',
    'waste hai': 'is a waste',
    'wahiyat': 'terrible',
    'ghatiya': 'awful/substandard',
    'dikkat aa rahi hai': 'having problems',
    'dikkat hai': 'there is a problem',
    'problem hai': 'there is a problem',
    'issue hai': 'there is an issue',
    'pareshaan hai': 'am troubled',
    'pareshan hai': 'am troubled',
    'tang aa gaye': 'fed up',
    'band karwa do': 'get it disconnected',
    'band karwa dunga': 'will get it disconnected',
    'hatwa dunga': 'will get it removed',

    # Time/frequency
    'roz roz': 'every day',
    'har roz': 'every day',
    'baar baar': 'again and again',
    'din mein': 'in a day',
    'raat ko': 'at night',
    'subah se': 'since morning',
    'shaam ko': 'in the evening',
    'kabhi kabhi': 'sometimes',
    'hamesha': 'always',
    'aksar': 'often',
}

# Sort by length (longest first) for greedy matching
SORTED_TRANSLATIONS = sorted(TRANSLATION_DICT.items(), key=lambda x: -len(x[0]))

def translate_comment(text):
    """Best-effort translation using dictionary matching."""
    if not isinstance(text, str):
        return str(text)

    result = text.lower()
    translations_applied = []

    for hindi, english in SORTED_TRANSLATIONS:
        if hindi in result:
            result = result.replace(hindi, english)
            translations_applied.append(hindi)

    return result.strip(), translations_applied

# Apply translation
print("  Translating comments...")
translations = df['oe_raw'].apply(translate_comment)
df['translated_comment'] = translations.apply(lambda x: x[0])
df['translations_applied'] = translations.apply(lambda x: x[1])
df['num_translations'] = df['translations_applied'].apply(len)

translated_count = (df['num_translations'] > 0).sum()
print(f"  Comments with translations applied: {translated_count} ({translated_count/len(df)*100:.1f}%)")

# ──────────────────────────────────────────────────────────────────────
# 4. Sentiment Analysis
# ──────────────────────────────────────────────────────────────────────
print("\n[4/8] Running sentiment analysis...")

NEGATIVE_WORDS = set([
    'kharab', 'bekar', 'ganda', 'slow', 'band', 'dikkat', 'problem', 'issue',
    'complaint', 'mahanga', 'mehnga', 'mehanga', 'costly', 'expensive',
    'disconnection', 'disconnect', 'down', 'waste', 'bakwas', 'chhota', 'khatam',
    'buffering', 'late', 'delay', 'rude', 'worst', 'pathetic', 'terrible',
    'useless', 'ghatiya', 'wahiyat', 'pareshaan', 'pareshan', 'tang',
    'nahi', 'nhi', 'nahin', 'nai', 'not', 'no', 'bad', 'poor', 'worst',
    'horrible', 'disappointed', 'frustrating', 'annoying', 'angry',
    'unresolved', 'pending', 'ignored', 'failed', 'failure',
    'dhoka', 'fraud', 'cheat', 'loot', 'unfair',
    'kam', 'low', 'less', 'chal nahi', 'nhi chal', 'stopped', 'broken',
    'hang', 'crash', 'error', 'drop', 'dropped', 'fluctuating', 'unstable',
])

POSITIVE_WORDS = set([
    'accha', 'acha', 'achha', 'badhiya', 'badiya', 'sahi', 'theek', 'thik',
    'fast', 'good', 'great', 'best', 'affordable', 'sasta', 'perfect',
    'excellent', 'smooth', 'superb', 'amazing', 'wonderful', 'reliable',
    'stable', 'consistent', 'helpful', 'quick', 'mast', 'maza', 'khush',
    'happy', 'satisfied', 'nice', 'fantastic', 'awesome', 'brilliant',
    'love', 'recommend', 'value', 'worth', 'comfortable', 'convenient',
    'fine', 'okay', 'ok', 'decent', 'proper', 'working', 'works',
    'resolved', 'fixed', 'solved', 'improvement', 'improved',
    'speed', 'clear', 'clean', 'neat',  # contextual - often positive in ISP
])

INTENSITY_AMPLIFIERS = set([
    'bahut', 'bht', 'bohot', 'bohut', 'bhut', 'boht', 'very', 'extremely',
    'bilkul', 'ekdum', 'totally', 'completely', 'absolutely', 'highly',
    'zyada', 'itna', 'kitna', 'really', 'truly', 'so', 'too', 'much',
    'hamesha', 'always', 'constantly', 'continuously', 'every', 'daily',
    'roz', 'har', 'baar baar', 'multiple', 'several', 'many', 'worst',
])

INTENSITY_REDUCERS = set([
    'kabhi', 'sometimes', 'thoda', 'little', 'slightly', 'generally',
    'mostly', 'usually', 'often', 'occasionally', 'minor', 'small',
])

def analyze_sentiment(text, nps_score):
    """Rule-based sentiment analysis."""
    if not isinstance(text, str) or len(text.strip()) < 2:
        return 'neutral', 1, 'indifference', False

    words = set(re.findall(r'[a-zA-Z\u0900-\u097F]+', text.lower()))
    text_lower = text.lower()

    neg_count = len(words & NEGATIVE_WORDS)
    pos_count = len(words & POSITIVE_WORDS)
    amp_count = len(words & INTENSITY_AMPLIFIERS)
    red_count = len(words & INTENSITY_REDUCERS)

    # Also check for Devanagari negative/positive patterns
    devanagari_neg = sum(1 for p in ['खराब', 'बेकार', 'धीमा', 'स्लो', 'बंद', 'दिक्कत', 'समस्या',
                                      'महंगा', 'गंदा', 'नहीं', 'नही', 'बहुत गंदा', 'बहुत स्लो',
                                      'काम नहीं', 'चल नहीं', 'बंद हो'] if p in text)
    devanagari_pos = sum(1 for p in ['अच्छा', 'बढ़िया', 'सही', 'ठीक', 'तेज़', 'किफ़ायती',
                                      'मस्त', 'खुश', 'बहुत अच्छा', 'सबसे अच्छा',
                                      'बहुत बढ़िया', 'शानदार'] if p in text)

    neg_count += devanagari_neg
    pos_count += devanagari_pos

    # Determine polarity
    net = pos_count - neg_count
    if net > 0:
        polarity = 'positive'
    elif net < 0:
        polarity = 'negative'
    else:
        # Tie-break: use NPS score
        if nps_score >= 7:
            polarity = 'positive'
        elif nps_score <= 4:
            polarity = 'negative'
        else:
            polarity = 'neutral'

    # Intensity (1-5)
    total_sentiment_words = neg_count + pos_count
    base_intensity = min(total_sentiment_words, 3)  # 0-3 base
    intensity = base_intensity + min(amp_count, 2) - min(red_count, 1)
    intensity = max(1, min(5, intensity))

    # Emotion detection
    anger_words = words & {'angry', 'tang', 'dhoka', 'fraud', 'loot', 'cheat', 'wahiyat', 'bakwas', 'worst', 'pathetic', 'horrible'}
    frustration_words = words & {'dikkat', 'problem', 'issue', 'complaint', 'baar', 'again', 'still', 'pending', 'unresolved'}
    disappointment_words = words & {'disappointed', 'expected', 'promised', 'thought', 'pehle', 'tha'}
    delight_words = words & {'amazing', 'fantastic', 'awesome', 'love', 'best', 'excellent', 'superb', 'brilliant', 'wonderful'}
    satisfaction_words = words & {'good', 'nice', 'fine', 'okay', 'satisfied', 'happy', 'accha', 'acha', 'sahi', 'theek', 'badhiya'}

    if anger_words and polarity == 'negative':
        emotion = 'anger'
    elif frustration_words and polarity == 'negative':
        emotion = 'frustration'
    elif disappointment_words and polarity == 'negative':
        emotion = 'disappointment'
    elif delight_words and polarity == 'positive':
        emotion = 'delight'
    elif satisfaction_words and polarity == 'positive':
        emotion = 'satisfaction'
    elif polarity == 'neutral' or total_sentiment_words == 0:
        emotion = 'indifference'
    elif polarity == 'negative':
        emotion = 'frustration'  # default negative emotion
    else:
        emotion = 'satisfaction'  # default positive emotion

    # Score-sentiment mismatch
    mismatch = False
    if polarity == 'positive' and nps_score <= 3:
        mismatch = True
    elif polarity == 'negative' and nps_score >= 9:
        mismatch = True

    return polarity, intensity, emotion, mismatch

# Apply sentiment analysis
print("  Analyzing sentiment...")
sentiments = df.apply(lambda row: analyze_sentiment(row['translated_comment'], row['nps_score']), axis=1)
df['sentiment_polarity'] = sentiments.apply(lambda x: x[0])
df['sentiment_intensity'] = sentiments.apply(lambda x: x[1])
df['emotion'] = sentiments.apply(lambda x: x[2])
df['score_sentiment_mismatch'] = sentiments.apply(lambda x: x[3])

print(f"  Sentiment polarity distribution:")
print(df['sentiment_polarity'].value_counts().to_string())

# ──────────────────────────────────────────────────────────────────────
# 5. Comment Quality
# ──────────────────────────────────────────────────────────────────────
print("\n[5/8] Scoring comment quality...")

def score_quality(text):
    if not isinstance(text, str):
        return 'low'
    text = text.strip()
    words = text.split()
    word_count = len(words)

    # Low quality: very short or generic
    if word_count <= 3:
        return 'low'
    low_quality_exact = {'ok', 'good', 'bad', 'nice', 'fine', 'best', 'worst',
                         'accha', 'acha', 'kharab', 'theek', 'sahi', 'badhiya', 'bekar'}
    if text.lower().strip().rstrip('.!?') in low_quality_exact:
        return 'low'

    # High quality: specific details
    has_amount = bool(re.search(r'[₹$]\s*\d+|\d+\s*(rs|rupee|rupay|paisa|paise)', text, re.I))
    has_duration = bool(re.search(r'\d+\s*(day|din|month|mahina|hour|ghanta|week|hafta)', text, re.I))
    has_specific = bool(re.search(r'(complaint|ticket|call|recharge|install|plan|speed|outage|disconnect|mbps|gb)', text, re.I))

    if word_count >= 15 and (has_amount or has_duration or has_specific):
        return 'high'
    if word_count >= 25:
        return 'high'

    return 'medium'

df['comment_quality'] = df['oe_raw'].apply(score_quality)
print(f"  Quality distribution:")
print(df['comment_quality'].value_counts().to_string())

# ──────────────────────────────────────────────────────────────────────
# 6. Entity Extraction
# ──────────────────────────────────────────────────────────────────────
print("\n[6/8] Extracting entities...")

def extract_entities(text):
    if not isinstance(text, str):
        return {}

    entities = {}

    # Monetary amounts
    amounts = re.findall(r'[₹]\s*(\d+)|(\d+)\s*(?:rs|rupee|rupay|rupaiye)', text, re.I)
    amounts = [a[0] or a[1] for a in amounts if a[0] or a[1]]
    if amounts:
        entities['amounts'] = amounts

    # Durations
    durations = re.findall(r'(\d+)\s*(?:day|din|days)', text, re.I)
    if durations:
        entities['durations_days'] = durations
    months = re.findall(r'(\d+)\s*(?:month|mahina|mahine)', text, re.I)
    if months:
        entities['durations_months'] = months

    # Competitor mentions
    competitors = []
    text_lower = text.lower()
    if any(x in text_lower for x in ['jio', 'jiofiber', 'jio fiber']):
        competitors.append('JioFiber')
    if any(x in text_lower for x in ['airtel', 'xstream']):
        competitors.append('Airtel')
    if any(x in text_lower for x in ['bsnl']):
        competitors.append('BSNL')
    if any(x in text_lower for x in ['hathway', 'act fibernet', 'spectra', 'tikona', 'excitel']):
        competitors.append('Other ISP')
    if competitors:
        entities['competitors'] = competitors

    # 28-day plan mention
    if re.search(r'28\s*(?:day|din|days|दिन)', text, re.I) or '28' in text:
        entities['mentions_28day'] = True

    # Speed mentions
    speed = re.findall(r'(\d+)\s*(?:mbps|mb)', text, re.I)
    if speed:
        entities['speed_mentions'] = speed

    return entities

df['entities'] = df['oe_raw'].apply(extract_entities)
df['mentions_28day'] = df['entities'].apply(lambda x: x.get('mentions_28day', False))
df['mentions_competitor'] = df['entities'].apply(lambda x: 'competitors' in x)
df['competitor_names'] = df['entities'].apply(lambda x: ','.join(x.get('competitors', [])))
df['mentions_amount'] = df['entities'].apply(lambda x: 'amounts' in x)

print(f"  28-day plan mentions: {df['mentions_28day'].sum()}")
print(f"  Competitor mentions: {df['mentions_competitor'].sum()}")
print(f"  Amount mentions: {df['mentions_amount'].sum()}")

# ──────────────────────────────────────────────────────────────────────
# 7. Save processed comments
# ──────────────────────────────────────────────────────────────────────
print("\n[7/8] Saving processed comments...")

output_cols = [
    'phone_number', 'nps_score', 'nps_group', 'oe_raw', 'has_comment',
    'tenure_bucket', 'tenure_excel', 'tenure_days', 'city', 'churn_label', 'is_churned',
    'Sprint ID', 'sprint_num', 'Channel', 'first_time_wifi',
    'NPS Reason - Primary', 'Primary Category',
    'detected_language', 'translated_comment', 'num_translations',
    'sentiment_polarity', 'sentiment_intensity', 'emotion', 'score_sentiment_mismatch',
    'comment_quality',
    'mentions_28day', 'mentions_competitor', 'competitor_names', 'mentions_amount',
]
output_cols = [c for c in output_cols if c in df.columns]

df_out = df[output_cols].copy()
# Convert entities to string for CSV
df_out.to_csv(os.path.join(DATA_DIR, 'nps_comments_processed.csv'), index=False, encoding='utf-8-sig')
print(f"  Saved nps_comments_processed.csv ({len(df_out)} rows)")

# ──────────────────────────────────────────────────────────────────────
# 8. Summary Report
# ──────────────────────────────────────────────────────────────────────
print("\n[8/8] Generating summary report...")

report_lines = []
def rpt(line=""):
    report_lines.append(line)
    print(line)

rpt("=" * 70)
rpt("PHASE 1 — HINDI NLP PROCESSING REPORT")
rpt("=" * 70)

rpt(f"\nTotal comments processed: {len(df)}")

rpt("\n--- LANGUAGE DISTRIBUTION ---")
lang_dist = df['detected_language'].value_counts()
for lang, count in lang_dist.items():
    rpt(f"  {lang:25s}: {count:>5d} ({count/len(df)*100:5.1f}%)")

rpt("\n--- SENTIMENT BY NPS GROUP ---")
ct_sent = pd.crosstab(df['nps_group'], df['sentiment_polarity'], normalize='index') * 100
rpt(ct_sent.round(1).to_string())

rpt("\n--- EMOTION BY NPS GROUP ---")
ct_emo = pd.crosstab(df['nps_group'], df['emotion'], normalize='index') * 100
rpt(ct_emo.round(1).to_string())

rpt("\n--- SCORE-SENTIMENT MISMATCH ---")
mismatch_rate = df['score_sentiment_mismatch'].mean() * 100
rpt(f"  Overall mismatch rate: {mismatch_rate:.1f}%")
mismatch_by_group = df.groupby('nps_group')['score_sentiment_mismatch'].mean() * 100
rpt(mismatch_by_group.round(1).to_string())

rpt("\n--- COMMENT QUALITY BY NPS GROUP ---")
ct_qual = pd.crosstab(df['nps_group'], df['comment_quality'], normalize='index') * 100
rpt(ct_qual.round(1).to_string())

rpt("\n--- 28-DAY PLAN MENTIONS ---")
rpt(f"  Total: {df['mentions_28day'].sum()} ({df['mentions_28day'].mean()*100:.1f}%)")
rpt(f"  By NPS Group:")
m28 = df.groupby('nps_group')['mentions_28day'].mean() * 100
rpt(m28.round(1).to_string())

rpt("\n--- COMPETITOR MENTIONS ---")
rpt(f"  Total: {df['mentions_competitor'].sum()}")
comp_counts = df[df['mentions_competitor']]['competitor_names'].value_counts()
if len(comp_counts) > 0:
    rpt(comp_counts.to_string())

rpt("\n--- LANGUAGE x NPS GROUP ---")
ct_lang = pd.crosstab(df['nps_group'], df['detected_language'])
rpt(ct_lang.to_string())

rpt("\n--- SENTIMENT x TENURE BUCKET ---")
ct_ten_sent = pd.crosstab(df['tenure_bucket'], df['sentiment_polarity'], normalize='index') * 100
rpt(ct_ten_sent.round(1).to_string())

# Sample high-quality detractor comments
rpt("\n--- TOP 15 HIGH-QUALITY DETRACTOR COMMENTS ---")
det_high = df[(df['nps_group'] == 'Detractor') & (df['comment_quality'] == 'high')].head(15)
for i, (_, row) in enumerate(det_high.iterrows(), 1):
    rpt(f"\n  [{i}] NPS={row['nps_score']:.0f} | {row['detected_language']} | {row['emotion']} | Intensity={row['sentiment_intensity']}")
    rpt(f"      Original: {str(row['oe_raw'])[:200]}")
    rpt(f"      Translated: {str(row['translated_comment'])[:200]}")

rpt("\n--- TOP 10 HIGH-QUALITY PROMOTER COMMENTS ---")
pro_high = df[(df['nps_group'] == 'Promoter') & (df['comment_quality'] == 'high')].head(10)
for i, (_, row) in enumerate(pro_high.iterrows(), 1):
    rpt(f"\n  [{i}] NPS={row['nps_score']:.0f} | {row['detected_language']} | {row['emotion']} | Intensity={row['sentiment_intensity']}")
    rpt(f"      Original: {str(row['oe_raw'])[:200]}")
    rpt(f"      Translated: {str(row['translated_comment'])[:200]}")

# Churn rate by sentiment
rpt("\n--- CHURN RATE BY SENTIMENT POLARITY ---")
churn_sent = df.groupby('sentiment_polarity').agg(
    total=('is_churned', 'count'),
    churned=('is_churned', 'sum')
)
churn_sent['churn_rate'] = (churn_sent['churned'] / churn_sent['total'] * 100).round(1)
rpt(churn_sent.to_string())

rpt("\n--- CHURN RATE BY EMOTION ---")
churn_emo = df.groupby('emotion').agg(
    total=('is_churned', 'count'),
    churned=('is_churned', 'sum')
)
churn_emo['churn_rate'] = (churn_emo['churned'] / churn_emo['total'] * 100).round(1)
rpt(churn_emo.to_string())

# Most common primary reasons x sentiment
rpt("\n--- PRIMARY REASON x SENTIMENT (coded subset) ---")
coded = df[df['NPS Reason - Primary'].notna()]
if len(coded) > 0:
    ct_reason = pd.crosstab(coded['NPS Reason - Primary'], coded['sentiment_polarity'])
    ct_reason['total'] = ct_reason.sum(axis=1)
    ct_reason = ct_reason.sort_values('total', ascending=False).head(15)
    rpt(ct_reason.to_string())

# Save report
report_path = os.path.join(OUTPUT_DIR, 'phase1_hindi_nlp_report.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
rpt(f"\n  Report saved to {report_path}")

rpt("\n" + "=" * 70)
rpt("PHASE 1 COMPLETE")
rpt("=" * 70)
