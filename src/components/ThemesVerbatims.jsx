import { useState, useMemo } from 'react';
import { getVerbatims, getThemeChanges } from '../utils/npsCalculator.js';
import {
  THEME_LABELS, THEME_COLORS, NPS_COLORS, SPRINT_MONTH_MAP,
  SPRINT_TO_MONTH, SPRINT_TO_QUARTER, MONTH_ORDER, QUARTER_ORDER,
} from '../utils/constants.js';

const GRANULARITY_OPTIONS = [
  { value: 'sprint', label: 'Sprint' },
  { value: 'month', label: 'Month' },
  { value: 'quarter', label: 'Quarter' },
];

export default function ThemesVerbatims({ records, sprints }) {
  // ─── Period filter ───
  const [granularity, setGranularity] = useState('sprint');
  const [selectedPeriod, setSelectedPeriod] = useState('all');
  const [comparePeriod, setComparePeriod] = useState('none');

  // ─── Verbatim-level filters (shown at top of verbatims section) ───
  const [selectedTheme, setSelectedTheme] = useState('all');
  const [selectedBand, setSelectedBand] = useState('all');
  const [searchText, setSearchText] = useState('');

  // Available periods
  const periods = useMemo(() => {
    if (granularity === 'sprint') return sprints;
    if (granularity === 'month') return MONTH_ORDER.filter(m => records.some(r => r.month === m));
    return QUARTER_ORDER.filter(q => records.some(r => r.quarter === q));
  }, [records, sprints, granularity]);

  // Filter records by period
  const filtered = useMemo(() => {
    if (selectedPeriod === 'all') return records;
    if (granularity === 'sprint') return records.filter(r => r.sprint_id === selectedPeriod);
    if (granularity === 'month') return records.filter(r => r.month === selectedPeriod);
    return records.filter(r => r.quarter === selectedPeriod);
  }, [records, selectedPeriod, granularity]);

  const periodLabel = (p) => granularity === 'sprint' ? `${p} (${SPRINT_MONTH_MAP[p] || ''})` : p;

  // ─── Comment Rate Stats ───
  const commentStats = useMemo(() => {
    const total = filtered.length;
    const withFeedback = filtered.filter(r => r.feedback && r.feedback.trim().length > 5);
    const promoters = filtered.filter(r => r.score_band === 'promoter');
    const detractors = filtered.filter(r => r.score_band === 'detractor');
    const passives = filtered.filter(r => r.score_band === 'passive');
    const proWithFb = promoters.filter(r => r.feedback && r.feedback.trim().length > 5);
    const detWithFb = detractors.filter(r => r.feedback && r.feedback.trim().length > 5);
    const pasWithFb = passives.filter(r => r.feedback && r.feedback.trim().length > 5);
    return {
      total,
      withFeedback: withFeedback.length,
      feedbackPct: total > 0 ? Math.round(withFeedback.length / total * 100) : 0,
      promoterCount: promoters.length,
      promoterFb: proWithFb.length,
      promoterFbPct: promoters.length > 0 ? Math.round(proWithFb.length / promoters.length * 100) : 0,
      detractorCount: detractors.length,
      detractorFb: detWithFb.length,
      detractorFbPct: detractors.length > 0 ? Math.round(detWithFb.length / detractors.length * 100) : 0,
      passiveCount: passives.length,
      passiveFb: pasWithFb.length,
      passiveFbPct: passives.length > 0 ? Math.round(pasWithFb.length / passives.length * 100) : 0,
    };
  }, [filtered]);

  // ─── Theme breakdown split by promoter vs detractor ───
  const themesByBand = useMemo(() => {
    const themed = filtered.filter(r => r.theme);
    const promoters = themed.filter(r => r.score_band === 'promoter');
    const detractors = themed.filter(r => r.score_band === 'detractor');
    const allThemes = [...new Set(themed.map(r => r.theme))];

    const proWithFb = filtered.filter(r => r.score_band === 'promoter' && r.feedback && r.feedback.trim().length > 5);
    const detWithFb = filtered.filter(r => r.score_band === 'detractor' && r.feedback && r.feedback.trim().length > 5);
    const proBase = proWithFb.length || 1;
    const detBase = detWithFb.length || 1;

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

  // ─── Available themes (for verbatim filter dropdown) ───
  const availableThemes = useMemo(() => {
    return [...new Set(filtered.filter(r => r.theme).map(r => r.theme))].sort();
  }, [filtered]);

  // ─── Theme comparison ───
  const themeChanges = useMemo(() => {
    if (comparePeriod === 'none' || selectedPeriod === 'all') return null;
    const filterKey = granularity === 'sprint' ? 'sprint_id' : granularity === 'month' ? 'month' : 'quarter';
    const recs1 = records.filter(r => r[filterKey] === comparePeriod);
    const recs2 = records.filter(r => r[filterKey] === selectedPeriod);
    if (recs1.length === 0 || recs2.length === 0) return null;
    return getThemeChanges(recs1, recs2);
  }, [records, selectedPeriod, comparePeriod, granularity]);

  // ─── Verbatims with filters ───
  const verbatims = useMemo(() => {
    let base = filtered;
    if (selectedBand !== 'all') base = base.filter(r => r.score_band === selectedBand);
    if (selectedTheme !== 'all') base = base.filter(r => r.theme === selectedTheme);
    const results = getVerbatims(base, { limit: 40 });
    if (searchText.trim()) {
      const q = searchText.toLowerCase().trim();
      return results.filter(v => v.feedback && v.feedback.toLowerCase().includes(q));
    }
    return results;
  }, [filtered, selectedTheme, selectedBand, searchText]);

  return (
    <div>
      {/* ─── Period Filter Bar ─── */}
      <div className="filter-bar">
        <div className="filter-group">
          <span className="filter-label">View By</span>
          <div className="toggle-group">
            {GRANULARITY_OPTIONS.map(opt => (
              <button key={opt.value} className={`toggle-btn ${granularity === opt.value ? 'active' : ''}`}
                onClick={() => { setGranularity(opt.value); setSelectedPeriod('all'); setComparePeriod('none'); }}>
                {opt.label}
              </button>
            ))}
          </div>
        </div>
        <div className="filter-group">
          <span className="filter-label">Period</span>
          <select className="filter-select" value={selectedPeriod}
            onChange={e => { setSelectedPeriod(e.target.value); setComparePeriod('none'); }}>
            <option value="all">All</option>
            {periods.map(p => <option key={p} value={p}>{periodLabel(p)}</option>)}
          </select>
        </div>
        {selectedPeriod !== 'all' && (
          <div className="filter-group">
            <span className="filter-label">Compare With</span>
            <select className="filter-select" value={comparePeriod} onChange={e => setComparePeriod(e.target.value)}>
              <option value="none">None</option>
              {periods.filter(p => p !== selectedPeriod).map(p => <option key={p} value={p}>{periodLabel(p)}</option>)}
            </select>
          </div>
        )}
      </div>

      {/* ─── Comment Rate Cards ─── */}
      <div className="stats-row" style={{ marginBottom: 20 }}>
        <div className="stat-card">
          <div className="stat-label">Comments / Total</div>
          <div className="stat-value" style={{ color: 'var(--accent)' }}>{commentStats.feedbackPct}%</div>
          <div className="stat-sub">{commentStats.withFeedback.toLocaleString()} of {commentStats.total.toLocaleString()} responded</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Detractor Comment Rate</div>
          <div className="stat-value" style={{ color: NPS_COLORS.detractor }}>{commentStats.detractorFbPct}%</div>
          <div className="stat-sub">{commentStats.detractorFb.toLocaleString()} of {commentStats.detractorCount.toLocaleString()} detractors</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Promoter Comment Rate</div>
          <div className="stat-value" style={{ color: NPS_COLORS.promoter }}>{commentStats.promoterFbPct}%</div>
          <div className="stat-sub">{commentStats.promoterFb.toLocaleString()} of {commentStats.promoterCount.toLocaleString()} promoters</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Passive Comment Rate</div>
          <div className="stat-value" style={{ color: NPS_COLORS.passive }}>{commentStats.passiveFbPct}%</div>
          <div className="stat-sub">{commentStats.passiveFb.toLocaleString()} of {commentStats.passiveCount.toLocaleString()} passives</div>
        </div>
      </div>

      {/* ─── Themes: Detractor vs Promoter ─── */}
      <div className="grid-2" style={{ marginBottom: 20 }}>
        <div className="card">
          <div className="card-title" style={{ color: NPS_COLORS.detractor }}>
            Detractor Themes
            <span style={{ fontWeight: 400, fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>% of detractors who commented</span>
          </div>
          {themesByBand.negativeThemes.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No detractor themes found</div>
          ) : (
            <table className="data-table">
              <thead><tr><th>Theme</th><th>% of Det.</th><th>Count</th></tr></thead>
              <tbody>
                {themesByBand.negativeThemes.slice(0, 10).map(t => (
                  <tr key={t.theme} style={{ cursor: 'pointer' }}
                    onClick={() => { setSelectedTheme(t.theme); setSelectedBand('detractor'); }}>
                    <td>
                      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: t.color, marginRight: 8 }} />
                      {t.label}
                    </td>
                    <td style={{ fontWeight: 600 }}>{t.pct}%</td>
                    <td style={{ color: 'var(--text-secondary)' }}>{t.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        <div className="card">
          <div className="card-title" style={{ color: NPS_COLORS.promoter }}>
            Promoter Themes
            <span style={{ fontWeight: 400, fontSize: 11, color: 'var(--text-muted)', marginLeft: 8 }}>% of promoters who commented</span>
          </div>
          {themesByBand.positiveThemes.length === 0 ? (
            <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>No promoter themes found</div>
          ) : (
            <table className="data-table">
              <thead><tr><th>Theme</th><th>% of Pro.</th><th>Count</th></tr></thead>
              <tbody>
                {themesByBand.positiveThemes.slice(0, 10).map(t => (
                  <tr key={t.theme} style={{ cursor: 'pointer' }}
                    onClick={() => { setSelectedTheme(t.theme); setSelectedBand('promoter'); }}>
                    <td>
                      <span style={{ display: 'inline-block', width: 8, height: 8, borderRadius: '50%', background: t.color, marginRight: 8 }} />
                      {t.label}
                    </td>
                    <td style={{ fontWeight: 600 }}>{t.pct}%</td>
                    <td style={{ color: 'var(--text-secondary)' }}>{t.count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>

      {/* ─── Theme Changes (comparison mode) ─── */}
      {themeChanges && themeChanges.length > 0 && (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-title">Theme Changes: {comparePeriod} → {selectedPeriod}</div>
          <table className="data-table">
            <thead><tr><th>Theme</th><th>Previous %</th><th>Current %</th><th>Change</th><th>Direction</th></tr></thead>
            <tbody>
              {themeChanges.slice(0, 10).map(c => (
                <tr key={c.theme}>
                  <td style={{ fontWeight: 500 }}>{THEME_LABELS[c.theme] || c.theme}</td>
                  <td>{c.pctPrev}%</td>
                  <td>{c.pctCurr}%</td>
                  <td style={{ fontWeight: 600, color: c.pctDelta > 0 ? NPS_COLORS.detractor : c.pctDelta < 0 ? NPS_COLORS.promoter : 'inherit' }}>
                    {c.pctDelta > 0 ? '+' : ''}{c.pctDelta}pp
                  </td>
                  <td>
                    {c.countPrev === 0 && c.countCurr > 0 ? <span className="badge detractor">Emerging</span>
                      : c.countPrev > 0 && c.countCurr === 0 ? <span className="badge promoter">Resolved</span>
                      : Math.abs(c.pctDelta) > 2 ? <span className={`badge ${c.pctDelta > 0 ? 'detractor' : 'promoter'}`}>{c.pctDelta > 0 ? 'Growing' : 'Shrinking'}</span>
                      : <span className="badge passive">Stable</span>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* ─── Verbatims with own filters at top ─── */}
      <div className="card">
        {/* Verbatim filter row — NPS type, theme, search */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center', marginBottom: 14, paddingBottom: 14, borderBottom: '1px solid var(--border)' }}>
          <div className="filter-group" style={{ margin: 0 }}>
            <span className="filter-label">NPS Type</span>
            <div className="toggle-group">
              {['all', 'promoter', 'passive', 'detractor'].map(band => (
                <button key={band}
                  className={`toggle-btn ${selectedBand === band ? 'active' : ''}`}
                  onClick={() => setSelectedBand(band)}
                  style={band !== 'all' && selectedBand === band ? { background: NPS_COLORS[band], borderColor: NPS_COLORS[band], color: 'white' } : {}}>
                  {band === 'all' ? 'All' : band.charAt(0).toUpperCase() + band.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div className="filter-group" style={{ margin: 0 }}>
            <span className="filter-label">Theme</span>
            <select className="filter-select" value={selectedTheme} onChange={e => setSelectedTheme(e.target.value)}>
              <option value="all">All Themes</option>
              {availableThemes.map(t => <option key={t} value={t}>{THEME_LABELS[t] || t}</option>)}
            </select>
          </div>

          {(selectedTheme !== 'all' || selectedBand !== 'all') && (
            <button
              style={{ fontSize: 11, color: 'var(--wiom-pink)', background: 'none', border: '1px solid var(--wiom-pink)', borderRadius: 'var(--radius-sm)', cursor: 'pointer', padding: '4px 8px' }}
              onClick={() => { setSelectedTheme('all'); setSelectedBand('all'); setSearchText(''); }}>
              Clear
            </button>
          )}

          <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-end' }}>
            <input
              type="text"
              placeholder="Search verbatims..."
              value={searchText}
              onChange={e => setSearchText(e.target.value)}
              style={{
                padding: '6px 12px',
                border: '1px solid var(--border)',
                borderRadius: 'var(--radius-sm)',
                fontSize: 12,
                fontFamily: "'Noto Sans', 'Noto Sans Devanagari', sans-serif",
                width: 220,
                background: 'white',
              }}
            />
          </div>
        </div>

        {/* Title with count */}
        <div className="card-title" style={{ marginBottom: 12 }}>
          Verbatims
          {selectedTheme !== 'all' && ` — ${THEME_LABELS[selectedTheme] || selectedTheme}`}
          {selectedBand !== 'all' && ` — ${selectedBand}s`}
          <span style={{ fontWeight: 400, fontSize: 12, color: 'var(--text-secondary)', marginLeft: 8 }}>
            ({verbatims.length} shown)
          </span>
        </div>

        {verbatims.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)', fontSize: 13, marginTop: 8 }}>
            No feedback matching the current filters. Try broadening your selection.
          </p>
        ) : (
          verbatims.map((v, i) => (
            <div key={i} className={`verbatim ${v.score_band}`}>
              <div className="verbatim-meta">
                <span className={`badge ${v.score_band}`}>NPS {v.score}</span>
                {' '}&middot;{' '}
                {THEME_LABELS[v.theme] || v.nps_reason_primary || 'No theme'}
                {' '}&middot;{' '}{v.sprint_id}
                {' '}&middot;{' '}{v.source}
              </div>
              <div className="verbatim-text">{v.feedback}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
