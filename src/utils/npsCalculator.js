/**
 * NPS calculation engine — ported from user-insights-agents/dashboard/nps_utils.py
 */

import {
  REASON_TO_THEME, SPRINT_TO_MONTH, SPRINT_TO_QUARTER,
  SOURCE_NORMALIZE, COLUMN_ALIASES,
} from './constants.js';

// ─── Score Classification ────────────────────────────────────────
export function classifyScore(score) {
  if (score >= 9) return 'promoter';
  if (score >= 7) return 'passive';
  return 'detractor';
}

// ─── Tenure Bucketing (3-tier, install date = day 1) ─────────────
export function assignTenureCut(tenureDays) {
  const td = parseInt(tenureDays, 10);
  if (isNaN(td) || td <= 0) return 'Unknown';
  if (td <= 60) return 'Early (1-2 mo)';
  if (td <= 180) return 'Mid (3-6 mo)';
  return 'Long (6+ mo)';
}

// ─── Source Normalization ────────────────────────────────────────
export function normalizeSource(raw) {
  if (!raw) return 'Unknown';
  const key = raw.toString().trim().toLowerCase();
  return SOURCE_NORMALIZE[key] || raw.toString().trim();
}

// ─── Flexible Column Matching ────────────────────────────────────
export function findColumn(headers, fieldName) {
  const aliases = COLUMN_ALIASES[fieldName] || [fieldName];
  const lower = headers.map(h => h.toLowerCase().trim());
  for (const alias of aliases) {
    const idx = lower.indexOf(alias.toLowerCase());
    if (idx !== -1) return headers[idx];
  }
  return null;
}

// ─── CSV Row → Record ────────────────────────────────────────────
export function parseRow(row, columnMap) {
  const getVal = (field) => {
    const col = columnMap[field];
    return col ? (row[col] ?? '').toString().trim() : '';
  };

  // Score is required — skip if non-numeric
  const rawScore = parseInt(getVal('score'), 10);
  if (isNaN(rawScore)) return null;
  const score = Math.max(0, Math.min(10, rawScore));

  // Respondent ID: strip ".0" Excel artifact
  let respondentId = getVal('respondent_id').replace(/\.0$/, '');

  // City: filter "#REF!" and "unknown"
  let city = getVal('city');
  if (['#ref!', 'unknown', 'nan', ''].includes(city.toLowerCase())) city = '';

  // Tenure: handle negative as unknown
  let tenureDays = parseInt(getVal('tenure_days'), 10);
  if (isNaN(tenureDays) || tenureDays < 0) tenureDays = 0;

  const reason = getVal('nps_reason_primary');

  // tenure_cut: use pre-computed value from CSV if present, else derive from tenure_days
  const rawTenureCut = getVal('tenure_cut');
  const tenureCut = rawTenureCut && rawTenureCut !== 'Unknown' && rawTenureCut !== ''
    ? rawTenureCut
    : assignTenureCut(tenureDays);

  // first_time_user: 'Yes', 'No', or '' (only available in newer sprints)
  const firstTimeUser = getVal('first_time_user');

  return {
    respondent_id: respondentId,
    score,
    score_band: classifyScore(score),
    feedback: getVal('feedback'),
    nps_reason_primary: reason,
    nps_reason_secondary: getVal('nps_reason_secondary'),
    plan_type: getVal('plan_type'),
    city,
    tenure_days: tenureDays,
    tenure_cut: tenureCut,
    source: normalizeSource(getVal('source')),
    sprint_id: getVal('sprint_id'),
    sprint_start: getVal('sprint_start'),
    first_time_user: firstTimeUser,
    // Derived
    theme: REASON_TO_THEME[reason] || '',
    month: '',   // set after parsing
    quarter: '', // set after parsing
  };
}

// ─── Enrich records with month/quarter ───────────────────────────
export function enrichRecords(records) {
  return records.map(r => ({
    ...r,
    month: SPRINT_TO_MONTH[r.sprint_id] || '',
    quarter: SPRINT_TO_QUARTER[r.sprint_id] || '',
  }));
}

// ─── NPS Calculation with Multinomial CI ─────────────────────────
export function calcNPS(records) {
  const n = records.length;
  if (n === 0) {
    return { nps: 0, ciLower: 0, ciUpper: 0, n: 0,
             promoters: 0, passives: 0, detractors: 0, confidence: 'none' };
  }

  const promoters = records.filter(r => r.score >= 9).length;
  const passives = records.filter(r => r.score >= 7 && r.score <= 8).length;
  const detractors = records.filter(r => r.score <= 6).length;

  const pPro = promoters / n;
  const pDet = detractors / n;
  const nps = Math.round((pPro - pDet) * 1000) / 10; // 1 decimal

  // Multinomial variance of NPS estimator
  const varNPS = (pPro * (1 - pPro) + pDet * (1 - pDet) + 2 * pPro * pDet) / n;
  const se = Math.sqrt(varNPS) * 100;
  const z = 1.96; // 95% CI
  const ciLower = Math.round((nps - z * se) * 10) / 10;
  const ciUpper = Math.round((nps + z * se) * 10) / 10;
  const ciWidth = ciUpper - ciLower;

  let confidence;
  if (n >= 100 && ciWidth < 15) confidence = 'high';
  else if (n >= 30 && ciWidth < 25) confidence = 'medium';
  else confidence = 'low';

  return { nps, ciLower, ciUpper, n, promoters, passives, detractors, confidence };
}

// ─── Two-Proportion Z-Test for NPS Significance ──────────────────
export function npsSignificanceTest(stats1, stats2) {
  const { n: n1, promoters: pro1, detractors: det1, nps: nps1 } = stats1;
  const { n: n2, promoters: pro2, detractors: det2, nps: nps2 } = stats2;

  if (n1 === 0 || n2 === 0) {
    return { delta: 0, significant: false, interpretation: 'Insufficient data' };
  }

  const delta = Math.round((nps2 - nps1) * 10) / 10;
  const pPro1 = pro1 / n1, pDet1 = det1 / n1;
  const pPro2 = pro2 / n2, pDet2 = det2 / n2;

  const var1 = (pPro1 * (1 - pPro1) + pDet1 * (1 - pDet1) + 2 * pPro1 * pDet1) / n1;
  const var2 = (pPro2 * (1 - pPro2) + pDet2 * (1 - pDet2) + 2 * pPro2 * pDet2) / n2;

  const seDiff = Math.sqrt(var1 + var2) * 100;
  if (seDiff === 0) return { delta, significant: false, interpretation: 'No variance' };

  const zScore = Math.abs(delta) / seDiff;
  const pValue = 2 * (1 - 0.5 * (1 + erf(zScore / Math.sqrt(2))));
  const significant = pValue < 0.05;

  let interpretation;
  if (significant) {
    const direction = delta > 0 ? 'improved' : 'declined';
    interpretation = `NPS ${direction} by ${Math.abs(delta)} pts (p=${pValue.toFixed(3)}) — statistically significant`;
  } else {
    interpretation = `NPS changed by ${delta} pts (p=${pValue.toFixed(3)}) — not statistically significant`;
  }

  return { delta, zScore: Math.round(zScore * 100) / 100, pValue: Math.round(pValue * 10000) / 10000, significant, interpretation };
}

// ─── Error function approximation (for p-value calculation) ──────
function erf(x) {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
  return sign * y;
}

// ─── NPS by Group ────────────────────────────────────────────────
export function calcNPSByGroup(records, groupKey) {
  const groups = {};
  records.forEach(r => {
    const val = r[groupKey] || 'Unknown';
    if (!groups[val]) groups[val] = [];
    groups[val].push(r);
  });

  return Object.entries(groups)
    .filter(([name]) => name && name !== 'Unknown' && name !== '')
    .map(([name, recs]) => ({ group: name, ...calcNPS(recs) }))
    .sort((a, b) => b.n - a.n);
}

// ─── Sprint Sort Key ─────────────────────────────────────────────
export function sprintSortKey(label) {
  const s = label.replace('Sprint ', '');
  if (s.toUpperCase().startsWith('RSP')) {
    const num = parseInt(s.substring(3), 10);
    return [2, isNaN(num) ? 99 : num];
  }
  const num = parseInt(s, 10);
  if (!isNaN(num)) return [1, num];
  return [3, 0];
}

export function sortSprints(labels) {
  return [...labels].sort((a, b) => {
    const ka = sprintSortKey(a), kb = sprintSortKey(b);
    return ka[0] - kb[0] || ka[1] - kb[1];
  });
}

// ─── Direction Classification ────────────────────────────────────
export function classifyDirection(delta) {
  if (delta > 2) return 'improving';
  if (delta < -2) return 'declining';
  return 'stable';
}

// ─── Verbatim Selection (evocativeness-scored) ───────────────────
export function getVerbatims(records, { theme, scoreBand, limit = 20 } = {}) {
  let filtered = records.filter(r => r.feedback && r.feedback.trim().length > 5);

  if (theme) filtered = filtered.filter(r => r.theme === theme);
  if (scoreBand) filtered = filtered.filter(r => r.score_band === scoreBand);

  return filtered
    .map(r => ({
      ...r,
      _evocScore: Math.min(r.feedback.length, 300) + Math.abs(r.score - 5) * 20,
    }))
    .sort((a, b) => b._evocScore - a._evocScore)
    .slice(0, limit);
}

// ─── Theme Statistics ────────────────────────────────────────────
export function getThemeStats(records) {
  const themed = records.filter(r => r.theme);
  const counts = {};
  themed.forEach(r => {
    if (!counts[r.theme]) counts[r.theme] = { count: 0, totalScore: 0 };
    counts[r.theme].count++;
    counts[r.theme].totalScore += r.score;
  });

  return Object.entries(counts)
    .map(([theme, { count, totalScore }]) => ({
      theme,
      count,
      avgScore: Math.round((totalScore / count) * 10) / 10,
      pct: Math.round((count / records.length) * 1000) / 10,
    }))
    .sort((a, b) => b.count - a.count);
}

// ─── Theme Changes Between Two Groups ────────────────────────────
export function getThemeChanges(records1, records2) {
  const stats1 = getThemeStats(records1);
  const stats2 = getThemeStats(records2);

  const map1 = Object.fromEntries(stats1.map(s => [s.theme, s]));
  const map2 = Object.fromEntries(stats2.map(s => [s.theme, s]));
  const allThemes = new Set([...Object.keys(map1), ...Object.keys(map2)]);

  const changes = [];
  allThemes.forEach(t => {
    const prev = map1[t] || { count: 0, pct: 0 };
    const curr = map2[t] || { count: 0, pct: 0 };
    changes.push({
      theme: t,
      countPrev: prev.count,
      countCurr: curr.count,
      pctPrev: prev.pct,
      pctCurr: curr.pct,
      pctDelta: Math.round((curr.pct - prev.pct) * 10) / 10,
    });
  });

  return changes.sort((a, b) => Math.abs(b.pctDelta) - Math.abs(a.pctDelta));
}
