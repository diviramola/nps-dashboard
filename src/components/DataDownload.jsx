import { useState, useEffect, useMemo } from 'react';

const SPRINT_ORDER = [
  'sprint_1', 'sprint_2', 'sprint_3', 'sprint_4', 'sprint_5',
  'sprint_6', 'sprint_7', 'sprint_8', 'sprint_9', 'sprint_10',
  'sprint_11', 'sprint_12', 'sprint_13', 'sprint_14',
  'sprint_rsp1', 'sprint_rsp2', 'sprint_rsp3',
];

const SPRINT_LABELS = {
  sprint_1: "Sprint 1 — Jul-1H '25",
  sprint_2: "Sprint 2 — Jul-2H '25",
  sprint_3: "Sprint 3 — Aug-1H '25",
  sprint_4: "Sprint 4 — Aug-2H '25",
  sprint_5: "Sprint 5 — Sep-1H '25",
  sprint_6: "Sprint 6 — Sep-2H '25",
  sprint_7: "Sprint 7 — Oct-1H '25",
  sprint_8: "Sprint 8 — Oct-2H '25",
  sprint_9: "Sprint 9 — Nov-1H '25",
  sprint_10: "Sprint 10 — Nov-2H '25",
  sprint_11: "Sprint 11 — Dec-1H '25",
  sprint_12: "Sprint 12 — Dec-2H '25",
  sprint_13: "Sprint 13 — Jan-1H '26",
  sprint_14: "Sprint 14 — Jan-2H '26",
  sprint_rsp1: "RSP1 — Feb-1H '26",
  sprint_rsp2: "RSP2 — Feb-2H '26",
  sprint_rsp3: "RSP3 — Mar-1H '26",
};

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

export default function DataDownload({ records }) {
  const [fileMeta, setFileMeta] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch file metadata on mount
  useEffect(() => {
    async function loadMeta() {
      try {
        const manifestRes = await fetch('./data/manifest.json');
        if (!manifestRes.ok) { setLoading(false); return; }
        const fileNames = await manifestRes.json();

        const meta = [];
        for (const fileName of fileNames) {
          try {
            const res = await fetch(`./data/sprints/${fileName}`, { method: 'HEAD' });
            const size = parseInt(res.headers.get('content-length') || '0', 10);
            const key = fileName.replace('.csv', '');
            meta.push({ fileName, key, size });
          } catch {
            const key = fileName.replace('.csv', '');
            meta.push({ fileName, key, size: 0 });
          }
        }

        // Sort by sprint order
        meta.sort((a, b) => SPRINT_ORDER.indexOf(a.key) - SPRINT_ORDER.indexOf(b.key));
        setFileMeta(meta);
      } catch {
        // no manifest
      } finally {
        setLoading(false);
      }
    }
    loadMeta();
  }, []);

  // Record counts per sprint
  const sprintCounts = useMemo(() => {
    const counts = {};
    records.forEach(r => {
      const sid = (r.sprint_id || '').toLowerCase().replace(/\s+/g, '_');
      counts[sid] = (counts[sid] || 0) + 1;
    });
    return counts;
  }, [records]);

  const totalSize = fileMeta.reduce((sum, f) => sum + f.size, 0);

  const handleDownload = (fileName) => {
    const link = document.createElement('a');
    link.href = `./data/sprints/${fileName}`;
    link.download = fileName;
    link.click();
  };

  const handleDownloadAll = async () => {
    // Download each file sequentially
    for (const f of fileMeta) {
      handleDownload(f.fileName);
      await new Promise(r => setTimeout(r, 300));
    }
  };

  if (loading) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: 40 }}>
        <div style={{ color: 'var(--text-secondary)' }}>Loading file index...</div>
      </div>
    );
  }

  if (fileMeta.length === 0) {
    return (
      <div className="card" style={{ textAlign: 'center', padding: 40 }}>
        <div style={{ color: 'var(--text-secondary)' }}>No bundled data files found.</div>
      </div>
    );
  }

  return (
    <div>
      {/* Summary card */}
      <div className="card" style={{ marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <div className="card-title" style={{ marginBottom: 4 }}>Raw Sprint Data</div>
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              {fileMeta.length} sprint CSVs &middot; {formatBytes(totalSize)} total &middot; {records.length.toLocaleString()} responses
            </div>
          </div>
          <button
            onClick={handleDownloadAll}
            style={{
              padding: '8px 18px',
              background: 'var(--wiom-pink)',
              color: 'white',
              border: 'none',
              borderRadius: 'var(--radius-sm)',
              fontFamily: "'Noto Sans', sans-serif",
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
              transition: 'background 0.15s',
            }}
            onMouseOver={e => e.target.style.background = 'var(--wiom-pink-hover)'}
            onMouseOut={e => e.target.style.background = 'var(--wiom-pink)'}
          >
            Download All
          </button>
        </div>

        <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 12 }}>
          Each CSV contains columns: respondent_id, score, feedback, nps_reason_primary, nps_reason_secondary,
          plan_type, city, tenure_days, source, sprint_id, sprint_start
        </div>
      </div>

      {/* File list */}
      <div className="card">
        <table className="data-table">
          <thead>
            <tr>
              <th>Sprint</th>
              <th>File</th>
              <th style={{ textAlign: 'right' }}>Responses</th>
              <th style={{ textAlign: 'right' }}>Size</th>
              <th style={{ textAlign: 'center' }}></th>
            </tr>
          </thead>
          <tbody>
            {fileMeta.map((f) => {
              const count = sprintCounts[f.key] || 0;
              const label = SPRINT_LABELS[f.key] || f.key;
              return (
                <tr key={f.key}>
                  <td style={{ fontWeight: 500 }}>{label}</td>
                  <td style={{ fontSize: 12, color: 'var(--text-secondary)', fontFamily: 'monospace' }}>{f.fileName}</td>
                  <td style={{ textAlign: 'right', fontWeight: 500 }}>{count.toLocaleString()}</td>
                  <td style={{ textAlign: 'right', fontSize: 12, color: 'var(--text-secondary)' }}>{formatBytes(f.size)}</td>
                  <td style={{ textAlign: 'center' }}>
                    <button
                      onClick={() => handleDownload(f.fileName)}
                      style={{
                        padding: '4px 12px',
                        background: 'transparent',
                        color: 'var(--wiom-pink)',
                        border: '1px solid var(--wiom-pink)',
                        borderRadius: 'var(--radius-sm)',
                        fontFamily: "'Noto Sans', sans-serif",
                        fontSize: 11,
                        fontWeight: 600,
                        cursor: 'pointer',
                        transition: 'all 0.15s',
                      }}
                      onMouseOver={e => { e.target.style.background = 'var(--wiom-pink)'; e.target.style.color = 'white'; }}
                      onMouseOut={e => { e.target.style.background = 'transparent'; e.target.style.color = 'var(--wiom-pink)'; }}
                    >
                      Download
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
