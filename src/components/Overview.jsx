import { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { calcNPS, calcNPSByGroup } from '../utils/npsCalculator.js';
import {
  SPRINT_MONTH_MAP, SPRINT_SHORT_MAP, SPRINT_TO_MONTH, SPRINT_TO_QUARTER,
  MONTH_ORDER, QUARTER_ORDER, NPS_COLORS, THEME_LABELS, THEME_COLORS,
} from '../utils/constants.js';
import NPSCard from './NPSCard.jsx';

const GRANULARITY_OPTIONS = [
  { value: 'sprint', label: 'Sprint' },
  { value: 'month', label: 'Month' },
  { value: 'quarter', label: 'Quarter' },
];

function generateHealthBlurb(overall, trendData) {
  if (!overall || overall.n === 0) return '';
  const parts = [];

  if (overall.nps > 20) parts.push(`Overall NPS of +${overall.nps} signals strong customer satisfaction.`);
  else if (overall.nps > 0) parts.push(`Overall NPS of +${overall.nps} is slightly positive but has room to grow.`);
  else if (overall.nps > -15) parts.push(`Overall NPS of ${overall.nps} is in negative territory — more detractors than promoters.`);
  else parts.push(`Overall NPS of ${overall.nps} indicates significant dissatisfaction among respondents.`);

  const proPct = Math.round(overall.promoters / overall.n * 100);
  const detPct = Math.round(overall.detractors / overall.n * 100);
  parts.push(`${proPct}% are promoters (9–10) and ${detPct}% are detractors (0–6).`);

  if (trendData.length >= 2) {
    const last = trendData[trendData.length - 1];
    const prev = trendData[trendData.length - 2];
    const delta = Math.round((last.nps - prev.nps) * 10) / 10;
    if (delta > 0) parts.push(`The most recent period saw NPS rise by +${delta} pts vs the previous one.`);
    else if (delta < 0) parts.push(`The most recent period saw NPS drop by ${delta} pts vs the previous one.`);
    else parts.push(`NPS was flat compared to the previous period.`);
  }

  return parts.join(' ');
}

export default function Overview({ records, sprints }) {
  const [granularity, setGranularity] = useState('sprint');
  const [selectedPeriod, setSelectedPeriod] = useState('all');

  // Available periods based on granularity
  const periods = useMemo(() => {
    if (granularity === 'sprint') return sprints;
    if (granularity === 'month') return MONTH_ORDER.filter(m => records.some(r => r.month === m));
    return QUARTER_ORDER.filter(q => records.some(r => r.quarter === q));
  }, [records, sprints, granularity]);

  // Filter records based on period selection
  const filtered = useMemo(() => {
    if (selectedPeriod === 'all') return records;
    if (granularity === 'sprint') return records.filter(r => r.sprint_id === selectedPeriod);
    if (granularity === 'month') return records.filter(r => r.month === selectedPeriod);
    return records.filter(r => r.quarter === selectedPeriod);
  }, [records, selectedPeriod, granularity]);

  const overall = useMemo(() => calcNPS(filtered), [filtered]);

  const feedbackRate = useMemo(() =>
    filtered.length > 0
      ? Math.round(filtered.filter(r => r.feedback && r.feedback.trim().length > 5).length / filtered.length * 100)
      : 0
  , [filtered]);

  // Trend data for line chart (always uses full records)
  const trendData = useMemo(() => {
    if (granularity === 'sprint') {
      return sprints.map(s => {
        const recs = records.filter(r => r.sprint_id === s);
        const stats = calcNPS(recs);
        return { key: s, label: SPRINT_SHORT_MAP[s] || s, fullLabel: SPRINT_MONTH_MAP[s] || s, ...stats };
      });
    }
    if (granularity === 'month') {
      return MONTH_ORDER.filter(m => records.some(r => r.month === m)).map(m => {
        const recs = records.filter(r => r.month === m);
        const stats = calcNPS(recs);
        return { key: m, label: m.replace(/ '2[0-9]/, ''), fullLabel: m, ...stats };
      });
    }
    return QUARTER_ORDER.filter(q => records.some(r => r.quarter === q)).map(q => {
      const recs = records.filter(r => r.quarter === q);
      const stats = calcNPS(recs);
      return { key: q, label: q.split(' (')[0], fullLabel: q, ...stats };
    });
  }, [records, sprints, granularity]);

  // Latest period delta (no significance, just direction)
  const latestDelta = useMemo(() => {
    if (trendData.length < 2) return null;
    const curr = trendData[trendData.length - 1];
    const prev = trendData[trendData.length - 2];
    const delta = Math.round((curr.nps - prev.nps) * 10) / 10;
    return { delta, currLabel: curr.fullLabel, prevLabel: prev.fullLabel };
  }, [trendData]);

  // Channel breakdown (based on filtered)
  const channelBreakdown = useMemo(() => calcNPSByGroup(filtered, 'source'), [filtered]);

  // Top themes split by promoter vs detractor
  const themesByBand = useMemo(() => {
    const themed = filtered.filter(r => r.theme);
    const promoters = themed.filter(r => r.score_band === 'promoter');
    const detractors = themed.filter(r => r.score_band === 'detractor');

    const proWithFb = filtered.filter(r => r.score_band === 'promoter' && r.feedback && r.feedback.trim().length > 5);
    const detWithFb = filtered.filter(r => r.score_band === 'detractor' && r.feedback && r.feedback.trim().length > 5);
    const proBase = proWithFb.length || 1;
    const detBase = detWithFb.length || 1;

    const allThemes = [...new Set(themed.map(r => r.theme))];
    const positiveThemes = [];
    const negativeThemes = [];

    allThemes.forEach(t => {
      const proCount = promoters.filter(r => r.theme === t).length;
      const detCount = detractors.filter(r => r.theme === t).length;
      const label = THEME_LABELS[t] || t;
      const color = THEME_COLORS[t] || '#999';

      if (proCount >= detCount && proCount > 0) {
        positiveThemes.push({ theme: t, label, color, count: proCount, pct: Math.round(proCount / proBase * 100) });
      }
      if (detCount > proCount || (detCount > 0 && proCount === 0)) {
        negativeThemes.push({ theme: t, label, color, count: detCount, pct: Math.round(detCount / detBase * 100) });
      }
    });

    positiveThemes.sort((a, b) => b.pct - a.pct);
    negativeThemes.sort((a, b) => b.pct - a.pct);
    return { positiveThemes, negativeThemes };
  }, [filtered]);

  // Health blurb
  const healthBlurb = useMemo(() => generateHealthBlurb(overall, trendData), [overall, trendData]);

  const periodLabel = (p) => granularity === 'sprint' ? `${p} (${SPRINT_MONTH_MAP[p] || ''})` : p;

  return (
    <div>
      {/* Controls row */}
      <div className="filter-bar" style={{ marginBottom: 16 }}>
        <div className="filter-group">
          <span className="filter-label">View By</span>
          <div className="toggle-group">
            {GRANULARITY_OPTIONS.map(opt => (
              <button
                key={opt.value}
                className={`toggle-btn ${granularity === opt.value ? 'active' : ''}`}
                onClick={() => { setGranularity(opt.value); setSelectedPeriod('all'); }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <div className="filter-group">
          <span className="filter-label">Period</span>
          <select
            className="filter-select"
            value={selectedPeriod}
            onChange={e => setSelectedPeriod(e.target.value)}
          >
            <option value="all">All {granularity === 'sprint' ? 'Sprints' : granularity === 'month' ? 'Months' : 'Quarters'}</option>
            {periods.map(p => (
              <option key={p} value={p}>{periodLabel(p)}</option>
            ))}
          </select>
        </div>
        {latestDelta && selectedPeriod === 'all' && (
          <div style={{ fontSize: 12, color: 'var(--text-secondary)', alignSelf: 'flex-end', paddingBottom: 6 }}>
            Latest: {latestDelta.currLabel} &mdash;{' '}
            <span style={{ fontWeight: 600, color: latestDelta.delta > 0 ? NPS_COLORS.promoter : latestDelta.delta < 0 ? NPS_COLORS.detractor : 'var(--text-secondary)' }}>
              {latestDelta.delta > 0 ? '+' : ''}{latestDelta.delta.toFixed(1)} pts vs previous
            </span>
          </div>
        )}
      </div>

      {/* Health blurb */}
      {healthBlurb && (
        <div className="card" style={{ marginBottom: 16, padding: '14px 18px', borderLeft: '3px solid var(--wiom-pink)' }}>
          <div style={{ fontSize: 13, lineHeight: 1.65, color: 'var(--text)' }}>{healthBlurb}</div>
        </div>
      )}

      {/* Stat cards */}
      <div className="stats-row">
        <NPSCard stats={overall} label={selectedPeriod === 'all' ? 'Overall NPS' : `NPS — ${selectedPeriod}`} />
        <div className="stat-card">
          <div className="stat-label">Responses</div>
          <div className="stat-value" style={{ color: 'var(--accent)' }}>{overall.n.toLocaleString()}</div>
          <div className="stat-sub">{periods.length} {granularity}s loaded</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Feedback Rate</div>
          <div className="stat-value" style={{ color: 'var(--accent)' }}>{feedbackRate}%</div>
          <div className="stat-sub">Responses with open text</div>
        </div>
        {latestDelta && selectedPeriod === 'all' && (
          <div className="stat-card">
            <div className="stat-label">Latest Change</div>
            <div className={`stat-value ${latestDelta.delta > 0 ? 'positive' : latestDelta.delta < 0 ? 'negative' : 'neutral'}`}>
              {latestDelta.delta > 0 ? '+' : ''}{latestDelta.delta.toFixed(1)}
            </div>
            <div className="stat-sub">{latestDelta.prevLabel} → {latestDelta.currLabel}</div>
          </div>
        )}
      </div>

      {/* NPS Trend Chart */}
      {trendData.length > 1 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-title">NPS Trend ({granularity})</div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trendData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="label" fontSize={10} interval={0} tickMargin={4} />
                <YAxis fontSize={11} domain={[-30, 30]} ticks={[-30, -20, -10, 0, 10, 20, 30]} />
                <Tooltip
                  labelFormatter={(label, payload) => payload?.[0]?.payload?.fullLabel || label}
                  formatter={(val, name) => {
                    if (name === 'nps') return [`${val > 0 ? '+' : ''}${Number(val).toFixed(1)}`, 'NPS'];
                    return [val, name];
                  }}
                  contentStyle={{ fontSize: 12 }}
                />
                <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                <Line
                  type="monotone" dataKey="nps" stroke="var(--accent)" strokeWidth={2.5}
                  dot={({ cx, cy, payload }) => {
                    const c = payload.nps > 5 ? NPS_COLORS.promoter : payload.nps < -5 ? NPS_COLORS.detractor : NPS_COLORS.passive;
                    return <circle key={`dot-${cx}-${cy}`} cx={cx} cy={cy} r={5} fill={c} stroke="white" strokeWidth={2} />;
                  }}
                  activeDot={{ r: 7 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* Channel Breakdown + Top Themes */}
      <div className="grid-2">
        {channelBreakdown.length > 0 && (
          <div className="card">
            <div className="card-title">NPS by Channel</div>
            <table className="data-table">
              <thead>
                <tr><th>Channel</th><th>NPS</th><th>n</th><th>Pro%</th><th>Pas%</th><th>Det%</th></tr>
              </thead>
              <tbody>
                {channelBreakdown.map(row => {
                  const proPct = row.n > 0 ? Math.round(row.promoters / row.n * 100) : 0;
                  const pasPct = row.n > 0 ? Math.round(row.passives / row.n * 100) : 0;
                  const detPct = row.n > 0 ? Math.round(row.detractors / row.n * 100) : 0;
                  return (
                    <tr key={row.group}>
                      <td style={{ fontWeight: 500 }}>{row.group}</td>
                      <td style={{ color: row.nps > 5 ? NPS_COLORS.promoter : row.nps < -5 ? NPS_COLORS.detractor : NPS_COLORS.passive, fontWeight: 600 }}>
                        {row.nps > 0 ? '+' : ''}{row.nps.toFixed(1)}
                      </td>
                      <td>{row.n.toLocaleString()}</td>
                      <td style={{ color: NPS_COLORS.promoter }}>{proPct}%</td>
                      <td style={{ color: NPS_COLORS.passive }}>{pasPct}%</td>
                      <td style={{ color: NPS_COLORS.detractor }}>{detPct}%</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Top Themes split by Promoter / Detractor */}
        <div className="card">
          <div className="card-title">Top Themes</div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: NPS_COLORS.promoter, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                Promoter Themes
              </div>
              {themesByBand.positiveThemes.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>None</div>
              ) : (
                themesByBand.positiveThemes.slice(0, 5).map(t => (
                  <div key={t.theme} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5, fontSize: 12 }}>
                    <span style={{ width: 7, height: 7, borderRadius: '50%', background: t.color, flexShrink: 0 }} />
                    <span style={{ flex: 1, color: 'var(--text)' }}>{t.label}</span>
                    <span style={{ fontWeight: 600, color: NPS_COLORS.promoter }}>{t.pct}%</span>
                  </div>
                ))
              )}
            </div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 600, color: NPS_COLORS.detractor, marginBottom: 6, textTransform: 'uppercase', letterSpacing: '0.4px' }}>
                Detractor Themes
              </div>
              {themesByBand.negativeThemes.length === 0 ? (
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>None</div>
              ) : (
                themesByBand.negativeThemes.slice(0, 5).map(t => (
                  <div key={t.theme} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 5, fontSize: 12 }}>
                    <span style={{ width: 7, height: 7, borderRadius: '50%', background: t.color, flexShrink: 0 }} />
                    <span style={{ flex: 1, color: 'var(--text)' }}>{t.label}</span>
                    <span style={{ fontWeight: 600, color: NPS_COLORS.detractor }}>{t.pct}%</span>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
