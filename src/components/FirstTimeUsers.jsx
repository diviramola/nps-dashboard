import { useState, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import { calcNPS } from '../utils/npsCalculator.js';
import {
  SPRINT_MONTH_MAP, MONTH_ORDER, QUARTER_ORDER,
  TENURE_CUT_ORDER, NPS_COLORS, THEME_LABELS,
} from '../utils/constants.js';
import NPSCard from './NPSCard.jsx';

const GRANULARITY_OPTIONS = [
  { value: 'sprint', label: 'Sprint' },
  { value: 'month',  label: 'Month' },
  { value: 'quarter', label: 'Quarter' },
];

const USER_TYPE_COLORS = {
  'First-time': '#3A8DD4',
  'Repeat':     '#D97FAF',
};

const MIN_N = 5;

export default function FirstTimeUsers({ records, sprints }) {
  const [granularity, setGranularity] = useState('sprint');
  const [selectedPeriod, setSelectedPeriod] = useState('all');

  // Base: only records that have first_time_user data
  const ftRecords = useMemo(() =>
    records.filter(r => r.first_time_user === 'Yes' || r.first_time_user === 'No'),
  [records]);

  // Periods that actually have FT data
  const activePeriods = useMemo(() => {
    if (granularity === 'sprint') return sprints.filter(s => ftRecords.some(r => r.sprint_id === s));
    if (granularity === 'month')  return MONTH_ORDER.filter(m => ftRecords.some(r => r.month === m));
    return QUARTER_ORDER.filter(q => ftRecords.some(r => r.quarter === q));
  }, [ftRecords, sprints, granularity]);

  // Period-filtered slice (drives all breakdown sections)
  const periodFiltered = useMemo(() => {
    if (selectedPeriod === 'all') return ftRecords;
    const key = granularity === 'sprint' ? 'sprint_id' : granularity === 'month' ? 'month' : 'quarter';
    return ftRecords.filter(r => r[key] === selectedPeriod);
  }, [ftRecords, selectedPeriod, granularity]);

  const firstTime = useMemo(() => periodFiltered.filter(r => r.first_time_user === 'Yes'), [periodFiltered]);
  const repeat    = useMemo(() => periodFiltered.filter(r => r.first_time_user === 'No'),  [periodFiltered]);

  // ── Trend chart (always full history, granularity-driven) ──────────────────
  const trendData = useMemo(() => {
    const key = granularity === 'sprint' ? 'sprint_id' : granularity === 'month' ? 'month' : 'quarter';
    return activePeriods.map(p => {
      const ft  = ftRecords.filter(r => r[key] === p && r.first_time_user === 'Yes');
      const rep = ftRecords.filter(r => r[key] === p && r.first_time_user === 'No');
      const ftS  = calcNPS(ft);
      const repS = calcNPS(rep);
      const label = granularity === 'sprint' ? (SPRINT_MONTH_MAP[p] || p) : p;
      return {
        period: p,
        label: label.length > 12 ? label.substring(0, 11) + '…' : label,
        fullLabel: label,
        highlighted: selectedPeriod !== 'all' && p === selectedPeriod,
        'First-time':   ftS.n >= MIN_N ? ftS.nps   : null,
        'First-time_n': ftS.n,
        'Repeat':       repS.n >= MIN_N ? repS.nps  : null,
        'Repeat_n':     repS.n,
      };
    });
  }, [ftRecords, activePeriods, granularity, selectedPeriod]);

  // ── Tenure breakdown ────────────────────────────────────────────────────────
  const tenureData = useMemo(() =>
    TENURE_CUT_ORDER.filter(t => periodFiltered.some(r => r.tenure_cut === t)).map(t => {
      const ft  = firstTime.filter(r => r.tenure_cut === t);
      const rep = repeat.filter(r => r.tenure_cut === t);
      const ftS  = calcNPS(ft);
      const repS = calcNPS(rep);
      return { tenure: t.split(' ')[0], fullLabel: t, ftS, repS };
    }),
  [periodFiltered, firstTime, repeat]);

  // ── City breakdown ──────────────────────────────────────────────────────────
  const cityData = useMemo(() => {
    const cities = [...new Set(periodFiltered.map(r => r.city).filter(Boolean))];
    return cities
      .map(city => {
        const ft  = firstTime.filter(r => r.city === city);
        const rep = repeat.filter(r => r.city === city);
        const ftS  = calcNPS(ft);
        const repS = calcNPS(rep);
        return { city, ftS, repS, total: ftS.n + repS.n };
      })
      .filter(d => d.total >= 10)
      .sort((a, b) => b.total - a.total)
      .slice(0, 8);
  }, [periodFiltered, firstTime, repeat]);

  // ── Theme breakdown ─────────────────────────────────────────────────────────
  const themeData = useMemo(() => {
    const themes = [...new Set(periodFiltered.map(r => r.theme).filter(Boolean))];
    return themes
      .map(theme => {
        const ftCount  = firstTime.filter(r => r.theme === theme).length;
        const repCount = repeat.filter(r => r.theme === theme).length;
        const ftPct  = firstTime.length > 0 ? Math.round(ftCount  / firstTime.length * 100) : 0;
        const repPct = repeat.length   > 0 ? Math.round(repCount / repeat.length   * 100) : 0;
        return {
          theme: THEME_LABELS[theme] || theme,
          'First-time': ftPct,
          'Repeat': repPct,
          total: ftCount + repCount,
        };
      })
      .filter(d => d.total >= 3)
      .sort((a, b) => b.total - a.total);
  }, [periodFiltered, firstTime, repeat]);

  // ── Verbatims ───────────────────────────────────────────────────────────────
  const ftVerbatims  = useMemo(() =>
    firstTime.filter(r => r.feedback && r.feedback.trim().length > 10)
      .sort((a, b) => (Math.abs(a.score - 5) < Math.abs(b.score - 5) ? 1 : -1))
      .slice(0, 8),
  [firstTime]);

  const repVerbatims = useMemo(() =>
    repeat.filter(r => r.feedback && r.feedback.trim().length > 10)
      .sort((a, b) => (Math.abs(a.score - 5) < Math.abs(b.score - 5) ? 1 : -1))
      .slice(0, 8),
  [repeat]);

  // ── Overall KPIs ────────────────────────────────────────────────────────────
  const overallAll = useMemo(() => calcNPS(periodFiltered), [periodFiltered]);
  const overallFT  = useMemo(() => calcNPS(firstTime),      [firstTime]);
  const overallRep = useMemo(() => calcNPS(repeat),         [repeat]);
  const npsDelta   = Math.round((overallFT.nps - overallRep.nps) * 10) / 10;

  if (ftRecords.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: 40, color: 'var(--text-secondary)' }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>📡</div>
        <div style={{ fontWeight: 600, fontSize: 15 }}>No first-time user data available</div>
        <div style={{ fontSize: 13, marginTop: 8 }}>
          This data is available from Sprint RSP5 onwards.
        </div>
      </div>
    );
  }

  const periodLabel = p => granularity === 'sprint' ? `${p} (${SPRINT_MONTH_MAP[p] || ''})` : p;

  const npsColor = nps => nps > 5 ? NPS_COLORS.promoter : nps < -5 ? NPS_COLORS.detractor : NPS_COLORS.passive;

  const NpsCell = ({ s }) => s.n < MIN_N
    ? <td>—</td>
    : (
      <td style={{ fontWeight: 600, color: npsColor(s.nps) }}>
        {s.nps > 0 ? '+' : ''}{s.nps.toFixed(1)}
        <span style={{ fontSize: 10, color: 'var(--text-muted)', marginLeft: 4 }}>n={s.n}</span>
      </td>
    );

  return (
    <div>
      {/* ── Filter bar ─────────────────────────────────────────────────────── */}
      <div className="card" style={{ marginBottom: 20, padding: '12px 18px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'flex-end' }}>
          <div className="filter-group">
            <span className="filter-label">View by</span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <div className="toggle-group">
                {GRANULARITY_OPTIONS.map(opt => (
                  <button key={opt.value}
                    className={`toggle-btn ${granularity === opt.value ? 'active' : ''}`}
                    onClick={() => { setGranularity(opt.value); setSelectedPeriod('all'); }}>
                    {opt.label}
                  </button>
                ))}
              </div>
              <select className="filter-select" value={selectedPeriod}
                onChange={e => setSelectedPeriod(e.target.value)}>
                <option value="all">All periods</option>
                {activePeriods.map(p => (
                  <option key={p} value={p}>{periodLabel(p)}</option>
                ))}
              </select>
            </div>
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', paddingBottom: 2 }}>
            {periodFiltered.length.toLocaleString()} responses
            {selectedPeriod !== 'all' && <span> · {periodLabel(selectedPeriod)}</span>}
            {' · '}First-time: {firstTime.length.toLocaleString()} · Repeat: {repeat.length.toLocaleString()}
          </div>
        </div>
      </div>

      {/* ── KPI cards ──────────────────────────────────────────────────────── */}
      <div className="stats-row" style={{ marginBottom: 20 }}>
        <NPSCard stats={overallAll} label={selectedPeriod === 'all' ? 'All (with WiFi history)' : `All · ${periodLabel(selectedPeriod)}`} />
        <NPSCard stats={overallFT}  label="First-time WiFi users" />
        <NPSCard stats={overallRep} label="Had WiFi before Wiom" />
        <div className="stat-card" style={{ textAlign: 'center' }}>
          <div className="stat-label">Delta (1st-time minus Repeat)</div>
          <div className="stat-value" style={{ color: npsDelta > 0 ? NPS_COLORS.promoter : npsDelta < 0 ? NPS_COLORS.detractor : NPS_COLORS.passive }}>
            {npsDelta > 0 ? '+' : ''}{npsDelta.toFixed(1)}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 4 }}>
            {npsDelta > 0 ? 'First-timers rate Wiom higher' : npsDelta < 0 ? 'Repeat users rate Wiom higher' : 'No difference'}
          </div>
        </div>
      </div>

      {/* ── Trend chart ────────────────────────────────────────────────────── */}
      {trendData.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 14 }}>
            <div className="card-title" style={{ margin: 0 }}>NPS Over Time — First-time vs Repeat</div>
            <div className="toggle-group">
              {GRANULARITY_OPTIONS.map(opt => (
                <button key={opt.value}
                  className={`toggle-btn ${granularity === opt.value ? 'active' : ''}`}
                  onClick={() => { setGranularity(opt.value); setSelectedPeriod('all'); }}>
                  {opt.label}
                </button>
              ))}
            </div>
          </div>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={trendData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                <XAxis dataKey="label" fontSize={10} interval={0} />
                <YAxis fontSize={11} domain={[-60, 60]} />
                <Tooltip
                  labelFormatter={(label, payload) => payload?.[0]?.payload?.fullLabel || label}
                  formatter={(val, name) => val === null ? ['n < 5', name] : [`${val > 0 ? '+' : ''}${Number(val).toFixed(1)}`, name]}
                  contentStyle={{ fontSize: 12 }}
                />
                <Legend verticalAlign="top" height={28} />
                <ReferenceLine y={0} stroke="#999" strokeDasharray="3 3" />
                <Bar dataKey="First-time" name="First-time WiFi" fill={USER_TYPE_COLORS['First-time']} radius={[3,3,0,0]} />
                <Bar dataKey="Repeat"     name="Had WiFi before" fill={USER_TYPE_COLORS['Repeat']}     radius={[3,3,0,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          {selectedPeriod !== 'all' && (
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
              Chart shows full history. Breakdowns below are filtered to {periodLabel(selectedPeriod)}.
            </div>
          )}
        </div>
      )}

      {/* ── Tenure + City ──────────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20, marginBottom: 20 }}>
        {tenureData.length > 0 && (
          <div className="card">
            <div className="card-title">NPS by Tenure</div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Tenure</th>
                  <th style={{ color: USER_TYPE_COLORS['First-time'] }}>First-time</th>
                  <th style={{ color: USER_TYPE_COLORS['Repeat'] }}>Repeat</th>
                  <th>Delta</th>
                </tr>
              </thead>
              <tbody>
                {tenureData.map(d => {
                  const delta = d.ftS.n >= MIN_N && d.repS.n >= MIN_N
                    ? Math.round((d.ftS.nps - d.repS.nps) * 10) / 10 : null;
                  return (
                    <tr key={d.tenure}>
                      <td style={{ fontWeight: 500 }}>{d.fullLabel}</td>
                      <NpsCell s={d.ftS} />
                      <NpsCell s={d.repS} />
                      <td style={{ fontWeight: 600, color: delta === null ? 'var(--text-muted)' : npsColor(delta) }}>
                        {delta !== null ? `${delta > 0 ? '+' : ''}${delta.toFixed(1)}` : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {cityData.length > 0 && (
          <div className="card">
            <div className="card-title">NPS by City</div>
            <table className="data-table">
              <thead>
                <tr>
                  <th>City</th>
                  <th style={{ color: USER_TYPE_COLORS['First-time'] }}>First-time</th>
                  <th style={{ color: USER_TYPE_COLORS['Repeat'] }}>Repeat</th>
                  <th>Delta</th>
                </tr>
              </thead>
              <tbody>
                {cityData.map(d => {
                  const delta = d.ftS.n >= MIN_N && d.repS.n >= MIN_N
                    ? Math.round((d.ftS.nps - d.repS.nps) * 10) / 10 : null;
                  return (
                    <tr key={d.city}>
                      <td style={{ fontWeight: 500 }}>{d.city}</td>
                      <NpsCell s={d.ftS} />
                      <NpsCell s={d.repS} />
                      <td style={{ fontWeight: 600, color: delta === null ? 'var(--text-muted)' : npsColor(delta) }}>
                        {delta !== null ? `${delta > 0 ? '+' : ''}${delta.toFixed(1)}` : '—'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ── Theme breakdown ─────────────────────────────────────────────────── */}
      {themeData.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-title">Top Themes — First-time vs Repeat (% of responses)</div>
          <div className="chart-container" style={{ height: 260 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={themeData.slice(0, 8)}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 140, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#eee" horizontal={false} />
                <XAxis type="number" fontSize={10} tickFormatter={v => `${v}%`} />
                <YAxis type="category" dataKey="theme" fontSize={11} width={135} />
                <Tooltip formatter={(val, name) => [`${val}%`, name]} contentStyle={{ fontSize: 12 }} />
                <Legend verticalAlign="top" height={28} />
                <Bar dataKey="First-time" name="First-time WiFi" fill={USER_TYPE_COLORS['First-time']} radius={[0,3,3,0]} />
                <Bar dataKey="Repeat"     name="Had WiFi before" fill={USER_TYPE_COLORS['Repeat']}     radius={[0,3,3,0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* ── Verbatims ──────────────────────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>
        <VerbatimPanel title="First-time WiFi Users" color={USER_TYPE_COLORS['First-time']} verbatims={ftVerbatims} />
        <VerbatimPanel title="Had WiFi Before Wiom"  color={USER_TYPE_COLORS['Repeat']}     verbatims={repVerbatims} />
      </div>
    </div>
  );
}

function VerbatimPanel({ title, color, verbatims }) {
  const [showAll, setShowAll] = useState(false);
  const shown = showAll ? verbatims : verbatims.slice(0, 4);

  return (
    <div className="card">
      <div className="card-title" style={{ color }}>{title}</div>
      {shown.length === 0 ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No verbatims for this filter</div>
      ) : (
        shown.map((r, i) => {
          const bc = r.score >= 9 ? NPS_COLORS.promoter : r.score <= 6 ? NPS_COLORS.detractor : NPS_COLORS.passive;
          const bg = r.score >= 9 ? NPS_COLORS.promoter_light : r.score <= 6 ? NPS_COLORS.detractor_light : NPS_COLORS.passive_light;
          return (
            <div key={i} style={{
              marginBottom: 10, padding: '8px 10px',
              borderLeft: `3px solid ${bc}`,
              background: bg,
              borderRadius: '0 4px 4px 0',
            }}>
              <div style={{
                fontSize: 12, lineHeight: 1.5, color: 'var(--text-primary)',
                fontFamily: "'Noto Sans', 'Noto Sans Devanagari', sans-serif",
              }}>
                {r.feedback}
              </div>
              <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginTop: 5, display: 'flex', gap: 10 }}>
                <span style={{ fontWeight: 700, color: bc }}>Score: {r.score}</span>
                {r.tenure_cut && r.tenure_cut !== 'Unknown' && <span>{r.tenure_cut}</span>}
                {r.city && <span>{r.city}</span>}
                {r.nps_reason_primary && <span>{r.nps_reason_primary}</span>}
              </div>
            </div>
          );
        })
      )}
      {verbatims.length > 4 && (
        <button onClick={() => setShowAll(!showAll)} style={{
          marginTop: 6, fontSize: 12, color, background: 'none',
          border: 'none', cursor: 'pointer', padding: 0, fontWeight: 600,
        }}>
          {showAll ? '▲ Show less' : `▼ Show ${verbatims.length - 4} more`}
        </button>
      )}
    </div>
  );
}
