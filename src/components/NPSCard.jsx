import { NPS_COLORS } from '../utils/constants.js';

export default function NPSCard({ stats, label = 'NPS' }) {
  const { nps, n, promoters, passives, detractors } = stats;

  const color = nps > 5 ? NPS_COLORS.promoter
    : nps < -5 ? NPS_COLORS.detractor
    : NPS_COLORS.passive;

  const proPct = n > 0 ? Math.round(promoters / n * 100) : 0;
  const pasPct = n > 0 ? Math.round(passives / n * 100) : 0;
  const detPct = n > 0 ? Math.round(detractors / n * 100) : 0;

  return (
    <div className="stat-card">
      <div className="stat-label">{label}</div>
      <div className="stat-value" style={{ color }}>{nps > 0 ? '+' : ''}{nps.toFixed(1)}</div>
      <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>
        n = {n.toLocaleString()}
      </div>
      {/* Stacked band bar */}
      <div style={{ display: 'flex', height: 6, borderRadius: 3, overflow: 'hidden', marginTop: 8, gap: 1 }}>
        <div style={{ width: `${proPct}%`, background: NPS_COLORS.promoter }} title={`Promoters ${proPct}%`} />
        <div style={{ width: `${pasPct}%`, background: NPS_COLORS.passive }} title={`Passives ${pasPct}%`} />
        <div style={{ width: `${detPct}%`, background: NPS_COLORS.detractor }} title={`Detractors ${detPct}%`} />
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 10, color: 'var(--text-secondary)', marginTop: 4 }}>
        <span style={{ color: NPS_COLORS.promoter }}>{proPct}% Pro</span>
        <span style={{ color: NPS_COLORS.passive }}>{pasPct}% Pas</span>
        <span style={{ color: NPS_COLORS.detractor }}>{detPct}% Det</span>
      </div>
    </div>
  );
}
