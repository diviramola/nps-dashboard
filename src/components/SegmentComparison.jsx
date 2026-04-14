import { useState, useMemo, useEffect } from 'react';
import { calcNPS } from '../utils/npsCalculator.js';
import {
  SPRINT_MONTH_MAP, SPRINT_SHORT_MAP, SPRINT_TO_MONTH, SPRINT_TO_QUARTER,
  MONTH_ORDER, QUARTER_ORDER, TENURE_CUT_ORDER, NPS_COLORS,
} from '../utils/constants.js';
import NPSCard from './NPSCard.jsx';

// ─── Dimension config ────────────────────────────────────────────────────────
const ROW_OPTIONS = [
  { value: 'tenure_cut',    label: 'Tenure' },
  { value: 'city',          label: 'City' },
  { value: 'source',        label: 'Channel' },
  { value: 'plan_type',     label: 'Plan Type' },
  { value: 'first_time_user', label: 'User Type' },
  { value: 'sprint_id',     label: 'Sprint' },
  { value: 'month',         label: 'Month' },
  { value: 'quarter',       label: 'Quarter' },
];

const COL_OPTIONS = [
  { value: 'sprint_id',     label: 'Sprint' },
  { value: 'month',         label: 'Month' },
  { value: 'quarter',       label: 'Quarter' },
  { value: 'tenure_cut',    label: 'Tenure' },
  { value: 'city',          label: 'City' },
  { value: 'source',        label: 'Channel' },
  { value: 'plan_type',     label: 'Plan Type' },
  { value: 'first_time_user', label: 'User Type' },
];

const GRANULARITY_OPTIONS = [
  { value: 'sprint', label: 'Sprint' },
  { value: 'month',  label: 'Month' },
  { value: 'quarter', label: 'Quarter' },
];

const ALL_CHANNELS = ['Call', 'CleverTap', 'WhatsApp'];

const MIN_N = 5; // suppress cells below this threshold

// ─── Color helpers ────────────────────────────────────────────────────────────
function npsBackground(nps, n) {
  if (n < MIN_N) return '#f0f0f0';
  const c = Math.max(-60, Math.min(60, nps));
  if (c >= 0) {
    const t = c / 60;
    return `rgb(${Math.round(245 - t * 199)},${Math.round(245 - t * 87)},${Math.round(245 - t * 151)})`;
  }
  const t = Math.abs(c) / 60;
  return `rgb(${Math.round(245 - t * 37)},${Math.round(245 - t * 181)},${Math.round(245 - t * 181)})`;
}

function npsTextColor(nps, n) {
  if (n < MIN_N) return '#aaa';
  return Math.abs(nps) > 35 ? '#fff' : '#222';
}

// ─── Sort dimension values ────────────────────────────────────────────────────
function sortDimValues(dim, vals, records) {
  if (dim === 'sprint_id') {
    const allSprints = Object.keys(SPRINT_MONTH_MAP);
    return vals.slice().sort((a, b) => {
      const ia = allSprints.indexOf(a), ib = allSprints.indexOf(b);
      return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
    });
  }
  if (dim === 'month')    return MONTH_ORDER.filter(m => vals.includes(m));
  if (dim === 'quarter')  return QUARTER_ORDER.filter(q => vals.includes(q));
  if (dim === 'tenure_cut') return TENURE_CUT_ORDER.filter(t => vals.includes(t));
  return vals.slice().sort();
}

function shortLabel(dim, val) {
  if (dim === 'sprint_id') return SPRINT_SHORT_MAP[val] || val.replace('Sprint ', '');
  return val;
}

// ─── Legend component ─────────────────────────────────────────────────────────
function HeatmapLegend() {
  const stops = [-60, -40, -20, 0, 20, 40, 60];
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 10, color: 'var(--text-secondary)', marginLeft: 'auto' }}>
      <span>NPS</span>
      <div style={{ display: 'flex', borderRadius: 3, overflow: 'hidden', border: '1px solid #e0e0e0' }}>
        {stops.map(v => (
          <div key={v} style={{
            width: 28, height: 16, background: npsBackground(v, 99),
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 9, color: npsTextColor(v, 99), fontWeight: 500,
          }}>
            {v > 0 ? '+' : ''}{v}
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function SegmentComparison({ records, sprints, sources }) {
  const [rowDim, setRowDim]   = useState('tenure_cut');
  const [colDim, setColDim]   = useState('sprint_id');
  const [granularity, setGranularity] = useState('sprint');
  const [selectedPeriod, setSelectedPeriod] = useState('all');
  const [selectedChannels, setSelectedChannels] = useState([...ALL_CHANNELS]);
  const [selectedTenure, setSelectedTenure] = useState([...TENURE_CUT_ORDER]);
  const [sortRows, setSortRows] = useState('natural'); // 'natural' | 'nps_desc' | 'nps_asc'
  const [showRowTotals, setShowRowTotals] = useState(true);

  const periods = useMemo(() => {
    if (granularity === 'sprint') return sprints;
    if (granularity === 'month') return MONTH_ORDER.filter(m => records.some(r => r.month === m));
    return QUARTER_ORDER.filter(q => records.some(r => r.quarter === q));
  }, [records, sprints, granularity]);

  const availableChannels = useMemo(() => {
    const found = [...new Set(records.map(r => r.source))].filter(Boolean);
    return ALL_CHANNELS.filter(c => found.includes(c)).concat(found.filter(c => !ALL_CHANNELS.includes(c)));
  }, [records]);

  const availableTenures = useMemo(() => {
    return TENURE_CUT_ORDER.filter(t => records.some(r => r.tenure_cut === t));
  }, [records]);

  useEffect(() => {
    setSelectedTenure([...TENURE_CUT_ORDER]);
  }, [selectedPeriod, granularity]);

  // Period filter
  const periodFiltered = useMemo(() => {
    if (selectedPeriod === 'all') return records;
    const key = granularity === 'sprint' ? 'sprint_id' : granularity === 'month' ? 'month' : 'quarter';
    return records.filter(r => r[key] === selectedPeriod);
  }, [records, selectedPeriod, granularity]);

  // Full filter
  const filtered = useMemo(() => {
    let f = periodFiltered;
    if (selectedChannels.length > 0 && selectedChannels.length < availableChannels.length) {
      f = f.filter(r => selectedChannels.includes(r.source));
    }
    const validTenures = selectedTenure.filter(t => availableTenures.includes(t));
    if (availableTenures.length > 0 && validTenures.length > 0 && validTenures.length < availableTenures.length) {
      f = f.filter(r => validTenures.includes(r.tenure_cut));
    }
    return f;
  }, [periodFiltered, selectedChannels, selectedTenure, availableChannels, availableTenures]);

  const overallStats = useMemo(() => calcNPS(filtered), [filtered]);

  const toggleItem = (list, setList, item) => {
    setList(prev => prev.includes(item) ? prev.filter(x => x !== item) : [...prev, item]);
  };

  // Build heatmap matrix
  const heatmap = useMemo(() => {
    const rawRowVals = [...new Set(filtered.map(r => r[rowDim]))].filter(v => v && v !== 'Unknown' && v !== '');
    const rawColVals = [...new Set(filtered.map(r => r[colDim]))].filter(v => v && v !== 'Unknown' && v !== '');

    const rowVals = sortDimValues(rowDim, rawRowVals, filtered);
    const colVals = sortDimValues(colDim, rawColVals, filtered);

    // For city, cap at top 12 by response count
    const trimmed = (dim, vals) => {
      if (dim !== 'city') return vals;
      return vals.slice(0, 12);
    };
    const rows = trimmed(rowDim, rowVals);
    const cols = trimmed(colDim, colVals);

    // Compute cell stats
    const cells = {};
    rows.forEach(rv => {
      cols.forEach(cv => {
        const recs = filtered.filter(r => r[rowDim] === rv && r[colDim] === cv);
        cells[`${rv}||${cv}`] = calcNPS(recs);
      });
    });

    // Row totals
    const rowTotals = {};
    rows.forEach(rv => {
      rowTotals[rv] = calcNPS(filtered.filter(r => r[rowDim] === rv));
    });

    // Col totals
    const colTotals = {};
    cols.forEach(cv => {
      colTotals[cv] = calcNPS(filtered.filter(r => r[colDim] === cv));
    });

    // Sort rows
    let sortedRows = [...rows];
    if (sortRows === 'nps_desc') sortedRows.sort((a, b) => (rowTotals[b]?.nps || -99) - (rowTotals[a]?.nps || -99));
    if (sortRows === 'nps_asc')  sortedRows.sort((a, b) => (rowTotals[a]?.nps || -99) - (rowTotals[b]?.nps || -99));

    return { rows: sortedRows, cols, cells, rowTotals, colTotals };
  }, [filtered, rowDim, colDim, sortRows]);

  const periodLabel = (p) => granularity === 'sprint' ? `${p} (${SPRINT_MONTH_MAP[p] || ''})` : p;

  const rowLabel  = ROW_OPTIONS.find(o => o.value === rowDim)?.label  || rowDim;
  const colLabel  = COL_OPTIONS.find(o => o.value === colDim)?.label  || colDim;

  return (
    <div>
      {/* ── Filter bar ─────────────────────────────────────────────────────── */}
      <div className="card" style={{ marginBottom: 20, padding: '14px 18px' }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'flex-start' }}>

          {/* Time period */}
          <div className="filter-group">
            <span className="filter-label">Time Period</span>
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
              <select className="filter-select" value={selectedPeriod} onChange={e => setSelectedPeriod(e.target.value)}>
                <option value="all">All periods</option>
                {periods.map(p => <option key={p} value={p}>{periodLabel(p)}</option>)}
              </select>
            </div>
          </div>

          {/* Tenure */}
          <div className="filter-group">
            <span className="filter-label">Tenure</span>
            {availableTenures.length === 0 ? (
              <span style={{ fontSize: 11, color: 'var(--text-muted)', alignSelf: 'center' }}>No data</span>
            ) : (
              <div className="chip-row">
                <button
                  className={`chip ${availableTenures.every(t => selectedTenure.includes(t)) ? 'selected' : ''}`}
                  onClick={() => setSelectedTenure([...TENURE_CUT_ORDER])}>All</button>
                {availableTenures.map(t => {
                  const allSelected = availableTenures.every(at => selectedTenure.includes(at));
                  return (
                    <button key={t}
                      className={`chip ${selectedTenure.includes(t) && !allSelected ? 'selected' : ''}`}
                      onClick={() => {
                        if (allSelected) setSelectedTenure([t]);
                        else toggleItem(selectedTenure, setSelectedTenure, t);
                      }}>
                      {t.split(' ')[0]}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Channel */}
          <div className="filter-group">
            <span className="filter-label">Channel</span>
            <div className="chip-row">
              <button
                className={`chip ${selectedChannels.length === availableChannels.length ? 'selected' : ''}`}
                onClick={() => setSelectedChannels([...availableChannels])}>All</button>
              {availableChannels.map(ch => (
                <button key={ch}
                  className={`chip ${selectedChannels.includes(ch) && selectedChannels.length < availableChannels.length ? 'selected' : ''}`}
                  onClick={() => {
                    if (selectedChannels.length === availableChannels.length) setSelectedChannels([ch]);
                    else toggleItem(selectedChannels, setSelectedChannels, ch);
                  }}>
                  {ch}
                </button>
              ))}
            </div>
          </div>

        </div>

        <div style={{ marginTop: 10, fontSize: 11, color: 'var(--text-secondary)' }}>
          {filtered.length.toLocaleString()} responses
          {selectedPeriod !== 'all' && <span> · {periodLabel(selectedPeriod)}</span>}
          {!availableTenures.every(t => selectedTenure.includes(t)) && (
            <span> · Tenure: {selectedTenure.filter(t => availableTenures.includes(t)).map(t => t.split(' ')[0]).join(', ')}</span>
          )}
          {selectedChannels.length < availableChannels.length && <span> · Channel: {selectedChannels.join(', ')}</span>}
        </div>
      </div>

      {/* ── Overall KPI ───────────────────────────────────────────────────── */}
      <div className="stats-row" style={{ marginBottom: 20 }}>
        <NPSCard stats={overallStats} label="Filtered NPS" />
      </div>

      {/* ── Heatmap ───────────────────────────────────────────────────────── */}
      <div className="card">
        {/* Heatmap header / controls */}
        <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 10, marginBottom: 14 }}>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>Compare</span>
          <select className="filter-select" value={rowDim}
            onChange={e => { setRowDim(e.target.value); if (e.target.value === colDim) setColDim(ROW_OPTIONS.find(o => o.value !== e.target.value)?.value || 'sprint_id'); }}>
            {ROW_OPTIONS.filter(o => o.value !== colDim).map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
          <span style={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>across</span>
          <select className="filter-select" value={colDim}
            onChange={e => { setColDim(e.target.value); if (e.target.value === rowDim) setRowDim(COL_OPTIONS.find(o => o.value !== e.target.value)?.value || 'tenure_cut'); }}>
            {COL_OPTIONS.filter(o => o.value !== rowDim).map(o => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>

          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginLeft: 8 }}>
            <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>Sort rows:</span>
            <select className="filter-select" style={{ fontSize: 11 }} value={sortRows} onChange={e => setSortRows(e.target.value)}>
              <option value="natural">Natural</option>
              <option value="nps_desc">NPS ↓</option>
              <option value="nps_asc">NPS ↑</option>
            </select>
          </div>

          <label style={{ display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, color: 'var(--text-secondary)', marginLeft: 8, cursor: 'pointer' }}>
            <input type="checkbox" checked={showRowTotals} onChange={e => setShowRowTotals(e.target.checked)} style={{ margin: 0 }} />
            Row totals
          </label>

          <HeatmapLegend />
        </div>

        {heatmap.rows.length === 0 || heatmap.cols.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '40px 0', color: 'var(--text-secondary)', fontSize: 13 }}>
            No data available for this dimension combination
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 11 }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', padding: '6px 10px', background: '#f8f8f8', borderBottom: '2px solid #e8e8e8', fontWeight: 600, fontSize: 11, color: 'var(--text-secondary)', minWidth: 110, position: 'sticky', left: 0, zIndex: 2 }}>
                    {rowLabel}
                  </th>
                  {heatmap.cols.map(cv => (
                    <th key={cv} style={{ padding: '5px 8px', background: '#f8f8f8', borderBottom: '2px solid #e8e8e8', fontWeight: 600, fontSize: 10, color: 'var(--text-secondary)', textAlign: 'center', minWidth: 68, whiteSpace: 'nowrap' }}>
                      {shortLabel(colDim, cv)}
                    </th>
                  ))}
                  {showRowTotals && (
                    <th style={{ padding: '5px 8px', background: '#f0f0f0', borderBottom: '2px solid #e8e8e8', fontWeight: 700, fontSize: 10, color: '#555', textAlign: 'center', minWidth: 68 }}>
                      Total
                    </th>
                  )}
                </tr>
              </thead>
              <tbody>
                {heatmap.rows.map((rv, ri) => (
                  <tr key={rv} style={{ background: ri % 2 === 0 ? '#fff' : '#fafafa' }}>
                    <td style={{ padding: '5px 10px', fontWeight: 600, fontSize: 11, color: '#333', borderRight: '1px solid #eee', position: 'sticky', left: 0, background: ri % 2 === 0 ? '#fff' : '#fafafa', zIndex: 1 }}>
                      {rv}
                    </td>
                    {heatmap.cols.map(cv => {
                      const s = heatmap.cells[`${rv}||${cv}`] || { nps: 0, n: 0 };
                      const bg = npsBackground(s.nps, s.n);
                      const fg = npsTextColor(s.nps, s.n);
                      return (
                        <td key={cv} style={{ padding: '4px 6px', textAlign: 'center', background: bg, border: '1px solid #f0f0f0' }}>
                          {s.n < MIN_N ? (
                            <span style={{ color: '#bbb', fontSize: 10 }}>—</span>
                          ) : (
                            <div>
                              <div style={{ fontWeight: 700, fontSize: 12, color: fg, lineHeight: 1.2 }}>
                                {s.nps > 0 ? '+' : ''}{s.nps.toFixed(0)}
                              </div>
                              <div style={{ fontSize: 9, color: s.n < MIN_N ? '#bbb' : fg, opacity: 0.75, lineHeight: 1 }}>
                                n={s.n}
                              </div>
                            </div>
                          )}
                        </td>
                      );
                    })}
                    {showRowTotals && (() => {
                      const s = heatmap.rowTotals[rv] || { nps: 0, n: 0 };
                      const bg = npsBackground(s.nps, s.n);
                      const fg = npsTextColor(s.nps, s.n);
                      return (
                        <td style={{ padding: '4px 6px', textAlign: 'center', background: bg, border: '1px solid #e4e4e4', borderLeft: '2px solid #ddd' }}>
                          {s.n < MIN_N ? (
                            <span style={{ color: '#bbb', fontSize: 10 }}>—</span>
                          ) : (
                            <div>
                              <div style={{ fontWeight: 700, fontSize: 12, color: fg, lineHeight: 1.2 }}>
                                {s.nps > 0 ? '+' : ''}{s.nps.toFixed(0)}
                              </div>
                              <div style={{ fontSize: 9, color: fg, opacity: 0.75, lineHeight: 1 }}>
                                n={s.n}
                              </div>
                            </div>
                          )}
                        </td>
                      );
                    })()}
                  </tr>
                ))}

                {/* Column totals row */}
                <tr style={{ background: '#f0f0f0', borderTop: '2px solid #ddd' }}>
                  <td style={{ padding: '5px 10px', fontWeight: 700, fontSize: 11, color: '#555', position: 'sticky', left: 0, background: '#f0f0f0', zIndex: 1 }}>
                    Total
                  </td>
                  {heatmap.cols.map(cv => {
                    const s = heatmap.colTotals[cv] || { nps: 0, n: 0 };
                    const bg = npsBackground(s.nps, s.n);
                    const fg = npsTextColor(s.nps, s.n);
                    return (
                      <td key={cv} style={{ padding: '4px 6px', textAlign: 'center', background: bg, border: '1px solid #e8e8e8' }}>
                        {s.n < MIN_N ? (
                          <span style={{ color: '#bbb', fontSize: 10 }}>—</span>
                        ) : (
                          <div>
                            <div style={{ fontWeight: 700, fontSize: 12, color: fg, lineHeight: 1.2 }}>
                              {s.nps > 0 ? '+' : ''}{s.nps.toFixed(0)}
                            </div>
                            <div style={{ fontSize: 9, color: fg, opacity: 0.75, lineHeight: 1 }}>
                              n={s.n}
                            </div>
                          </div>
                        )}
                      </td>
                    );
                  })}
                  {showRowTotals && (() => {
                    const s = overallStats;
                    const bg = npsBackground(s.nps, s.n);
                    const fg = npsTextColor(s.nps, s.n);
                    return (
                      <td style={{ padding: '4px 6px', textAlign: 'center', background: bg, border: '1px solid #ddd', borderLeft: '2px solid #ddd' }}>
                        <div>
                          <div style={{ fontWeight: 700, fontSize: 12, color: fg, lineHeight: 1.2 }}>
                            {s.nps > 0 ? '+' : ''}{s.nps.toFixed(0)}
                          </div>
                          <div style={{ fontSize: 9, color: fg, opacity: 0.75, lineHeight: 1 }}>
                            n={s.n}
                          </div>
                        </div>
                      </td>
                    );
                  })()}
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {heatmap.rows.length > 0 && heatmap.cols.length > 0 && (
          <div style={{ marginTop: 8, fontSize: 10, color: 'var(--text-muted)' }}>
            Cells with fewer than {MIN_N} responses are suppressed (—). City rows capped at top 12.
          </div>
        )}
      </div>

      {/* ── Row detail table ─────────────────────────────────────────────── */}
      {heatmap.rows.length > 0 && (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-title">{rowLabel} — NPS Summary</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>{rowLabel}</th>
                <th>NPS</th>
                <th>n</th>
                <th>Promoter %</th>
                <th>Passive %</th>
                <th>Detractor %</th>
                <th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {heatmap.rows.map(rv => {
                const s = heatmap.rowTotals[rv] || { nps: 0, n: 0, promoters: 0, passives: 0, detractors: 0, confidence: 'low' };
                const color = s.nps > 5 ? NPS_COLORS.promoter : s.nps < -5 ? NPS_COLORS.detractor : NPS_COLORS.passive;
                return (
                  <tr key={rv}>
                    <td style={{ fontWeight: 500 }}>{rv}</td>
                    <td style={{ fontWeight: 700, color }}>{s.nps > 0 ? '+' : ''}{s.nps.toFixed(1)}</td>
                    <td>{s.n.toLocaleString()}</td>
                    <td style={{ color: NPS_COLORS.promoter }}>{s.n > 0 ? Math.round(s.promoters / s.n * 100) : 0}%</td>
                    <td style={{ color: NPS_COLORS.passive }}>{s.n > 0 ? Math.round(s.passives / s.n * 100) : 0}%</td>
                    <td style={{ color: NPS_COLORS.detractor }}>{s.n > 0 ? Math.round(s.detractors / s.n * 100) : 0}%</td>
                    <td>
                      <span style={{
                        fontSize: 10, padding: '2px 6px', borderRadius: 10,
                        background: s.confidence === 'high' ? '#e8f5ee' : s.confidence === 'medium' ? '#fef6e0' : '#fdeaea',
                        color: s.confidence === 'high' ? NPS_COLORS.promoter : s.confidence === 'medium' ? '#c8960a' : NPS_COLORS.detractor,
                      }}>
                        {s.confidence}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
