/**
 * CSV parser using Papa Parse — handles Wiom NPS sprint CSVs.
 * Supports flexible column matching and BOM-encoded files.
 */

import Papa from 'papaparse';
import { findColumn, parseRow, enrichRecords } from './npsCalculator.js';

/**
 * Parse a single CSV file (File object or string content).
 * Returns { records: NPSRecord[], errors: string[], fileName: string }
 */
export function parseCSVFile(file) {
  return new Promise((resolve) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (h) => h.replace(/^\uFEFF/, '').trim(), // Strip BOM
      complete: (results) => {
        const headers = results.meta.fields || [];
        const errors = [];

        // Build column map
        const columnMap = {};
        const fields = ['score', 'feedback', 'respondent_id', 'nps_reason_primary',
          'nps_reason_secondary', 'plan_type', 'city', 'tenure_days', 'source',
          'sprint_id', 'sprint_start', 'tenure_cut', 'first_time_user'];

        fields.forEach(field => {
          const col = findColumn(headers, field);
          if (col) columnMap[field] = col;
        });

        if (!columnMap.score) {
          errors.push(`No score column found in headers: ${headers.join(', ')}`);
          resolve({ records: [], errors, fileName: file.name || 'unknown' });
          return;
        }

        const records = [];
        let skipped = 0;

        results.data.forEach((row) => {
          const rec = parseRow(row, columnMap);
          if (rec) {
            records.push(rec);
          } else {
            skipped++;
          }
        });

        if (skipped > 0) {
          errors.push(`Skipped ${skipped} rows with invalid/missing scores`);
        }

        resolve({
          records: enrichRecords(records),
          errors,
          fileName: file.name || 'unknown',
        });
      },
      error: (err) => {
        resolve({ records: [], errors: [err.message], fileName: file.name || 'unknown' });
      },
    });
  });
}

/**
 * Parse multiple CSV files and merge records.
 * Returns { allRecords: NPSRecord[], fileResults: FileResult[] }
 */
export async function parseMultipleCSVs(files) {
  const fileResults = [];
  const allRecords = [];

  for (const file of files) {
    const result = await parseCSVFile(file);
    fileResults.push(result);
    allRecords.push(...result.records);
  }

  return { allRecords: enrichRecords(allRecords), fileResults };
}

/**
 * Load a CSV from a URL (for bundled data files).
 */
export async function loadCSVFromURL(url) {
  const response = await fetch(url);
  const text = await response.text();
  return new Promise((resolve) => {
    const fakeFile = new File([text], url.split('/').pop(), { type: 'text/csv' });
    parseCSVFile(fakeFile).then(resolve);
  });
}
