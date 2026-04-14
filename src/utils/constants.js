/**
 * Constants and mappings — ported from user-insights-agents/dashboard/nps_utils.py
 */

// Sprint → human-readable month label
export const SPRINT_MONTH_MAP = {
  'Sprint 1': "Jul-1H '25", 'Sprint 2': "Jul-2H '25",
  'Sprint 3': "Aug-1H '25", 'Sprint 4': "Aug-2H '25",
  'Sprint 5': "Sep-1H '25", 'Sprint 6': "Sep-2H '25",
  'Sprint 7': "Oct-1H '25", 'Sprint 8': "Oct-2H '25",
  'Sprint 9': "Nov-1H '25", 'Sprint 10': "Nov-2H '25",
  'Sprint 11': "Dec-1H '25", 'Sprint 12': "Dec-2H '25",
  'Sprint 13': "Jan-1H '26", 'Sprint 14': "Jan-2H '26",
  'Sprint RSP1': "Feb-1H '26", 'Sprint RSP2': "Feb-2H '26",
  'Sprint RSP3': "Mar-1H '26", 'Sprint RSP5': "Apr-1H '26",
};

// Short tick labels for chart x-axes (saves horizontal space)
export const SPRINT_SHORT_MAP = {
  'Sprint 1': 'Jul-1H', 'Sprint 2': 'Jul-2H',
  'Sprint 3': 'Aug-1H', 'Sprint 4': 'Aug-2H',
  'Sprint 5': 'Sep-1H', 'Sprint 6': 'Sep-2H',
  'Sprint 7': 'Oct-1H', 'Sprint 8': 'Oct-2H',
  'Sprint 9': 'Nov-1H', 'Sprint 10': 'Nov-2H',
  'Sprint 11': 'Dec-1H', 'Sprint 12': 'Dec-2H',
  'Sprint 13': 'Jan-1H', 'Sprint 14': 'Jan-2H',
  'Sprint RSP1': 'Feb-1H', 'Sprint RSP2': 'Feb-2H',
  'Sprint RSP3': 'Mar-1H', 'Sprint RSP5': 'Apr-1H',
};

// Sprint → month aggregation
export const SPRINT_TO_MONTH = {
  'Sprint 1': "Jul '25", 'Sprint 2': "Jul '25",
  'Sprint 3': "Aug '25", 'Sprint 4': "Aug '25",
  'Sprint 5': "Sep '25", 'Sprint 6': "Sep '25",
  'Sprint 7': "Oct '25", 'Sprint 8': "Oct '25",
  'Sprint 9': "Nov '25", 'Sprint 10': "Nov '25",
  'Sprint 11': "Dec '25", 'Sprint 12': "Dec '25",
  'Sprint 13': "Jan '26", 'Sprint 14': "Jan '26",
  'Sprint RSP1': "Feb '26", 'Sprint RSP2': "Feb '26",
  'Sprint RSP3': "Mar '26", 'Sprint RSP5': "Apr '26",
};

// Sprint → quarter aggregation
export const SPRINT_TO_QUARTER = {
  'Sprint 1': "Q1 FY26 (Jul-Sep '25)", 'Sprint 2': "Q1 FY26 (Jul-Sep '25)",
  'Sprint 3': "Q1 FY26 (Jul-Sep '25)", 'Sprint 4': "Q1 FY26 (Jul-Sep '25)",
  'Sprint 5': "Q1 FY26 (Jul-Sep '25)", 'Sprint 6': "Q1 FY26 (Jul-Sep '25)",
  'Sprint 7': "Q2 FY26 (Oct-Dec '25)", 'Sprint 8': "Q2 FY26 (Oct-Dec '25)",
  'Sprint 9': "Q2 FY26 (Oct-Dec '25)", 'Sprint 10': "Q2 FY26 (Oct-Dec '25)",
  'Sprint 11': "Q2 FY26 (Oct-Dec '25)", 'Sprint 12': "Q2 FY26 (Oct-Dec '25)",
  'Sprint 13': "Q3 FY26 (Jan-Mar '26)", 'Sprint 14': "Q3 FY26 (Jan-Mar '26)",
  'Sprint RSP1': "Q3 FY26 (Jan-Mar '26)", 'Sprint RSP2': "Q3 FY26 (Jan-Mar '26)",
  'Sprint RSP3': "Q3 FY26 (Jan-Mar '26)", 'Sprint RSP5': "Q1 FY27 (Apr '26+)",
};

export const QUARTER_ORDER = [
  "Q1 FY26 (Jul-Sep '25)", "Q2 FY26 (Oct-Dec '25)", "Q3 FY26 (Jan-Mar '26)",
  "Q1 FY27 (Apr '26+)",
];

export const MONTH_ORDER = [
  "Jul '25", "Aug '25", "Sep '25", "Oct '25", "Nov '25", "Dec '25",
  "Jan '26", "Feb '26", "Mar '26", "Apr '26",
];

// Pre-coded nps_reason_primary → theme mapping
export const REASON_TO_THEME = {
  'Slow Speed': 'speed_quality',
  'Good Speed': 'speed_quality',
  'Frequent Disconnections': 'reliability',
  'Internet Down': 'reliability',
  'RDNI (Recharge Done, No Internet)': 'reliability',
  'Affordable': 'pricing_value',
  'Expensive': 'pricing_value',
  '28-Day Plan Issue': 'pricing_value',
  'No/ Late Complaint Resolution': 'customer_support',
  'No/Late Complaint Resolution': 'customer_support',
  'Fast Complaint Resolution': 'customer_support',
  'Bad Customer Support - Call Centre': 'customer_support',
  'Good Customer Support - Call Centre': 'customer_support',
  'Bad Customer Support - Technician': 'technician_service',
  'Range Issue': 'network_coverage',
  'General Good Service': 'general_satisfaction',
  'General Bad Service': 'general_satisfaction',
  'Bad Application': 'content_usage',
  'OTT Request, No OTT complaint': 'content_usage',
  'Shifting not Feasible': 'installation',
};

export const THEME_LABELS = {
  speed_quality: 'Speed & Quality',
  reliability: 'Reliability & Uptime',
  pricing_value: 'Pricing & Value',
  customer_support: 'Customer Support',
  technician_service: 'Technician Service',
  network_coverage: 'Network Coverage',
  general_satisfaction: 'General Satisfaction',
  content_usage: 'Content & Usage',
  installation: 'Installation',
  billing_recharge: 'Billing & Recharge',
  churn_risk: 'Churn & Competitor',
  recommendation: 'Recommendation',
};

export const THEME_COLORS = {
  speed_quality: '#D04040',
  reliability: '#B53535',
  pricing_value: '#2E9E5E',
  customer_support: '#8A4DAD',
  technician_service: '#D97FAF',
  network_coverage: '#3A8DD4',
  general_satisfaction: '#8E9AA0',
  content_usage: '#E89820',
  installation: '#1AAD8C',
  billing_recharge: '#D94F7A',
  churn_risk: '#D04040',
  recommendation: '#2E9E5E',
};

export const NPS_COLORS = {
  promoter: '#2E9E5E',
  passive: '#E8B818',
  detractor: '#D04040',
  promoter_light: '#E8F5EE',
  passive_light: '#FEF6E0',
  detractor_light: '#FDEAEA',
};

// Source normalization map
export const SOURCE_NORMALIZE = {
  'wa': 'WhatsApp', 'whatsapp': 'WhatsApp', 'in - wa': 'WhatsApp', 'g- form in wa': 'WhatsApp',
  'ct': 'CleverTap', 'clevertap': 'CleverTap',
  'call': 'Call',
};

export const TENURE_CUT_ORDER = ['Early (1-2 mo)', 'Mid (3-6 mo)', 'Long (6+ mo)'];

// Column name aliases for flexible CSV header matching
export const COLUMN_ALIASES = {
  score: ['score', 'nps_score', 'nps', 'rating', 'nps_rating'],
  feedback: ['feedback', 'open_text', 'comment', 'comments', 'verbatim', 'text', 'response', 'open_ended', 'why', 'oe'],
  respondent_id: ['respondent_id', 'id', 'user_id', 'customer_id', 'phone', 'mobile', 'phone number'],
  nps_reason_primary: ['nps_reason_primary', 'nps reason - primary', 'primary_reason', 'reason'],
  nps_reason_secondary: ['nps_reason_secondary', 'nps reason - secondary', 'secondary_reason'],
  plan_type: ['plan_type', 'plan', 'segment', 'product'],
  city: ['city', 'location', 'geography', 'region'],
  tenure_days: ['tenure_days', 'tenure', 'days_since_activation'],
  source: ['source', 'channel', 'acquisition_source'],
  sprint_id: ['sprint_id', 'sprint', 'cycle'],
  sprint_start: ['sprint_start', 'date', 'timestamp', 'created_at', 'response_date'],
  tenure_cut: ['tenure_cut', 'tenure_bucket', 'tenure_segment'],
  first_time_user: ['first_time_user', 'first time user?', 'first_time', 'new_to_wifi'],
};
