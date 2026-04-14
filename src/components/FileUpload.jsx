import { useState, useRef, useCallback } from 'react';

export default function FileUpload({ onFilesLoaded, fileLog, hasData }) {
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const inputRef = useRef(null);

  const handleFiles = useCallback(async (files) => {
    const csvFiles = Array.from(files).filter(f =>
      f.name.endsWith('.csv') || f.type === 'text/csv'
    );
    if (csvFiles.length === 0) return;

    setLoading(true);
    try {
      await onFilesLoaded(csvFiles);
    } finally {
      setLoading(false);
    }
  }, [onFilesLoaded]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }, [handleFiles]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragOver(true);
  }, []);

  return (
    <div>
      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-title">Upload Sprint CSV Files</div>
        <div className="card-subtitle">
          Drop one or more NPS sprint CSVs. Each file should have a <code>score</code> column
          and optionally: feedback, nps_reason_primary, tenure_days, source, sprint_id.
        </div>

        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={() => setDragOver(false)}
          onClick={() => inputRef.current?.click()}
        >
          <div className="upload-icon">{loading ? '...' : '📁'}</div>
          <p>{loading ? 'Processing...' : 'Drop CSV files here or click to browse'}</p>
          <p style={{ fontSize: 12, marginTop: 4 }}>Accepts .csv files with flexible header matching</p>
        </div>

        <input
          ref={inputRef}
          type="file"
          accept=".csv"
          multiple
          style={{ display: 'none' }}
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>

      {/* Upload log */}
      {fileLog.length > 0 && (
        <div className="card">
          <div className="card-title">Upload History</div>
          <table className="data-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Records</th>
                <th>Status</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {fileLog.slice().reverse().map((entry, i) => (
                <tr key={i}>
                  <td style={{ fontWeight: 500 }}>{entry.name}</td>
                  <td>{entry.count.toLocaleString()}</td>
                  <td>
                    {entry.errors.length === 0 ? (
                      <span className="badge promoter">OK</span>
                    ) : (
                      <span className="badge detractor" title={entry.errors.join('; ')}>
                        {entry.errors.length} warning{entry.errors.length > 1 ? 's' : ''}
                      </span>
                    )}
                  </td>
                  <td style={{ color: 'var(--text-secondary)', fontSize: 12 }}>{entry.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!hasData && fileLog.length === 0 && (
        <div className="card" style={{ marginTop: 16 }}>
          <div className="card-title">Getting Started</div>
          <p style={{ fontSize: 13, color: 'var(--text-secondary)', lineHeight: 1.8 }}>
            Upload your Wiom NPS sprint CSV files to populate the dashboard.
            Each CSV should contain at minimum a <code>score</code> column (0–10).
            The parser auto-detects column names using flexible matching
            (e.g., "NPS", "rating", "nps_score" all map to score).
            <br /><br />
            Sprint files from <code>data/nps/raw/</code> in the user-insights-agents repo
            are compatible. Upload multiple sprints to enable trend analysis and comparison views.
          </p>
        </div>
      )}
    </div>
  );
}
