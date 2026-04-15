import { useState, useEffect, useMemo, useCallback } from 'react';
import { parseCSVFile, loadCSVFromURL } from './utils/csvParser.js';
import {
  calcNPS, calcNPSByGroup, sortSprints, npsSignificanceTest,
  getThemeStats, getVerbatims, getThemeChanges, classifyDirection,
} from './utils/npsCalculator.js';
import {
  SPRINT_MONTH_MAP, SPRINT_TO_MONTH, SPRINT_TO_QUARTER,
  MONTH_ORDER, QUARTER_ORDER, TENURE_CUT_ORDER,
  THEME_LABELS, THEME_COLORS, NPS_COLORS,
} from './utils/constants.js';
import ScoreTrends from './components/ScoreTrends.jsx';
import SegmentComparison from './components/SegmentComparison.jsx';
import ThemesVerbatims from './components/ThemesVerbatims.jsx';
import FirstTimeUsers from './components/FirstTimeUsers.jsx';
import FileUpload from './components/FileUpload.jsx';
import DataDownload from './components/DataDownload.jsx';
import Overview from './components/Overview.jsx';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'trends', label: 'Score Trends' },
  { id: 'segments', label: 'Segment Comparison' },
  { id: 'firsttime', label: 'First-time vs Repeat' },
  { id: 'themes', label: 'Themes & Verbatims' },
  { id: 'data', label: 'Data' },
  { id: 'upload', label: 'Upload' },
];

export default function App() {
  const [records, setRecords] = useState([]);
  const [activeTab, setActiveTab] = useState('overview');
  const [fileLog, setFileLog] = useState([]);
  const [loading, setLoading] = useState(true);
  const [darkMode, setDarkMode] = useState(() => {
    try { return window.matchMedia('(prefers-color-scheme: dark)').matches; } catch { return false; }
  });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
  }, [darkMode]);

  // Auto-load bundled sprint CSVs on startup
  useEffect(() => {
    async function loadBundledData() {
      try {
        const manifestRes = await fetch('./data/manifest.json');
        if (!manifestRes.ok) {
          setLoading(false);
          setActiveTab('upload');
          return;
        }
        const fileNames = await manifestRes.json();

        const allRecords = [];
        const log = [];

        for (const fileName of fileNames) {
          try {
            const result = await loadCSVFromURL(`./data/sprints/${fileName}`);
            allRecords.push(...result.records);
            log.push({
              name: fileName,
              count: result.records.length,
              errors: result.errors,
              time: 'bundled',
            });
          } catch (err) {
            log.push({ name: fileName, count: 0, errors: [err.message], time: 'bundled' });
          }
        }

        if (allRecords.length > 0) {
          setRecords(allRecords);
          setFileLog(log);
          setActiveTab('overview');
        } else {
          setActiveTab('upload');
        }
      } catch (err) {
        console.warn('No bundled data found, starting empty:', err.message);
        setActiveTab('upload');
      } finally {
        setLoading(false);
      }
    }

    loadBundledData();
  }, []);

  // Handle new files uploaded
  const handleFilesLoaded = useCallback(async (files) => {
    const newRecords = [];
    const newLog = [];

    for (const file of files) {
      const result = await parseCSVFile(file);
      newRecords.push(...result.records);
      newLog.push({
        name: file.name,
        count: result.records.length,
        errors: result.errors,
        time: new Date().toLocaleTimeString(),
      });
    }

    setRecords(prev => {
      // Merge: replace records from same sprint_id, add new ones
      const existing = new Map();
      prev.forEach(r => {
        const key = `${r.sprint_id}__${r.respondent_id}`;
        existing.set(key, r);
      });
      newRecords.forEach(r => {
        const key = `${r.sprint_id}__${r.respondent_id}`;
        existing.set(key, r);
      });
      return Array.from(existing.values());
    });

    setFileLog(prev => [...prev, ...newLog]);
    if (newRecords.length > 0) setActiveTab('overview');
  }, []);

  // Derived data
  const sprints = useMemo(() => {
    const unique = [...new Set(records.map(r => r.sprint_id))].filter(Boolean);
    return sortSprints(unique);
  }, [records]);

  const sources = useMemo(() => {
    return [...new Set(records.map(r => r.source))].filter(Boolean).sort();
  }, [records]);

  const hasData = records.length > 0;

  if (loading) {
    return (
      <div className="app">
        <header className="header">
          <h1>Wiom NPS Dashboard</h1>
          <div className="header-right">
            <button className="theme-toggle" onClick={() => setDarkMode(d => !d)} title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}>
              <span className="theme-toggle-icon">{darkMode ? '☀️' : '🌙'}</span>
              {darkMode ? 'Light' : 'Dark'}
            </button>
          </div>
        </header>
        <main className="main" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
          <div style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
            <div style={{ fontSize: 14, fontWeight: 600, color: 'var(--wiom-pink)', marginBottom: 12, letterSpacing: '0.5px', textTransform: 'uppercase' }}>Loading</div>
            <div style={{ fontSize: 15 }}>Parsing 18 sprint CSVs...</div>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <h1>Wiom NPS Dashboard</h1>
        <div className="header-right">
          {hasData && (
            <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              {records.length.toLocaleString()} responses &middot; {sprints.length} sprints
            </span>
          )}
          <button className="theme-toggle" onClick={() => setDarkMode(d => !d)} title={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}>
            <span className="theme-toggle-icon">{darkMode ? '☀️' : '🌙'}</span>
            {darkMode ? 'Light' : 'Dark'}
          </button>
        </div>
      </header>

      <main className="main">
        {/* Tab Navigation */}
        <nav className="tab-nav">
          {TABS.map(tab => (
            <button
              key={tab.id}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
              onClick={() => setActiveTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
        </nav>

        {/* Tab Content */}
        {activeTab === 'data' && (
          <DataDownload records={records} />
        )}

        {activeTab === 'upload' && (
          <FileUpload onFilesLoaded={handleFilesLoaded} fileLog={fileLog} hasData={hasData} />
        )}

        {activeTab === 'overview' && (
          hasData ? (
            <Overview records={records} sprints={sprints} />
          ) : (
            <EmptyState />
          )
        )}

        {activeTab === 'trends' && (
          hasData ? (
            <ScoreTrends records={records} sprints={sprints} />
          ) : (
            <EmptyState />
          )
        )}

        {activeTab === 'segments' && (
          hasData ? (
            <SegmentComparison records={records} sprints={sprints} sources={sources} />
          ) : (
            <EmptyState />
          )
        )}

        {activeTab === 'firsttime' && (
          hasData ? (
            <FirstTimeUsers records={records} sprints={sprints} />
          ) : (
            <EmptyState />
          )
        )}

        {activeTab === 'themes' && (
          hasData ? (
            <ThemesVerbatims records={records} sprints={sprints} />
          ) : (
            <EmptyState />
          )
        )}
      </main>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-icon">📊</div>
      <h3>No data loaded</h3>
      <p>Go to the Upload Data tab to load sprint CSV files.</p>
    </div>
  );
}
