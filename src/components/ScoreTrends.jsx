import { useState, useMemo } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend,
} from 'recharts';
import { calcNPS } from '../utils/npsCalculator.js';
import {
  SPRINT_MONTH_MAP, SPRINT_SHORT_MAP, SPRINT_TO_MONTH, SPRINT_TO_QUARTER,
  MONTH_ORDER, QUARTER_ORDER, NPS_COLORS,
} from '../utils/constants.js';

const GRANULARITY_OPTIONS = [
  { value: 'sprint', label: 'Sprint' },
  { value: 'month', label: 'Month' },
  { value: 'quarter', label: 'Quarter' },
];

const BAND_FILLS = {
  promoter: '#7ECFA0',
  passive: '#F5D76E',
  detractor: '#E88A8A',
};

function BarLabel({ x, y, width, height, value }) {
  if (height < 14 || value < 5) return null;
  return (
    <text x={x + width / 2} y={y + height / 2} fill="#fff" fontSize={10} fontWeight={600} textAnchor="middle" dominantBaseline="central">
      {value}%
    </text>
  );
}

export default function ScoreTrends({ records, sprints }) {
  const [granularity, setGranularity] = useState('sprint');

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

  const distData = useMemo(() => {
    return trendData.map(d => {
      const proPct = d.n > 0 ? Math.round(d.promoters / d.n * 100) : 0;
      const pasPct = d.n > 0 ? Math.round(d.passives / d.n * 100) : 0;
      const detPct = d.n > 0 ? Math.round(d.detractors / d.n * 100) : 0;
      return { label: d.label, fullLabel: d.fullLabel, promoter: proPct, passive: pasPct, detractor: detPct, n: d.n };
    });
  }, [trendData]);

  return (
    <div>
      <div style={{ display: 'flex', gap: 16, alignItems: 'center', marginBottom: 20 }}>
        <div className="filter-group">
          <span className="filter-label">Time Granularity</span>
          <div className="toggle-group">
            {GRANULARITY_OPTIONS.map(opt => (
              <button key={opt.value} className={`toggle-btn ${granularity === opt.value ? 'active' : ''}`} onClick={() => setGranularity(opt.value)}>
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title">NPS Score Over Time ({granularity})</div>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={trendData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
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
                  const color = payload.nps > 5 ? NPS_COLORS.promoter : payload.nps < -5 ? NPS_COLORS.detractor : NPS_COLORS.passive;
                  return <circle key={`dot-${cx}-${cy}`} cx={cx} cy={cy} r={5} fill={color} stroke="white" strokeWidth={2} />;
                }}
                activeDot={{ r: 7 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card">
        <div className="card-title">Score Band Distribution (%)</div>
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={distData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="label" fontSize={10} interval={0} tickMargin={4} />
              <YAxis fontSize={11} domain={[0, 100]} tickFormatter={v => `${v}%`} />
              <Tooltip
                labelFormatter={(label, payload) => payload?.[0]?.payload?.fullLabel || label}
                formatter={(val, name) => [`${val}%`, name.charAt(0).toUpperCase() + name.slice(1)]}
                contentStyle={{ fontSize: 12 }}
              />
              <Legend verticalAlign="top" height={30} />
              <Bar dataKey="promoter" stackId="a" fill={BAND_FILLS.promoter} name="Promoter" label={<BarLabel />} />
              <Bar dataKey="passive" stackId="a" fill={BAND_FILLS.passive} name="Passive" label={<BarLabel />} />
              <Bar dataKey="detractor" stackId="a" fill={BAND_FILLS.detractor} name="Detractor" label={<BarLabel />} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
