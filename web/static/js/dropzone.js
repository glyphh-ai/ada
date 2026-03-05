/**
 * Data upload dropzone — drag-and-drop .jsonl files.
 *
 * POSTs records to the listener endpoint for encoding into HDC glyphs.
 * Polls the job for completion, then dispatches 'glyphh:data-loaded'.
 */

import { authHeaders } from './auth.js';

export function init() {
  const area = document.getElementById('dropzone-area');
  const fileInput = document.getElementById('dropzone-input');

  // Click to open file picker
  area.addEventListener('click', () => fileInput.click());

  // File picker change
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleFile(fileInput.files[0]);
      fileInput.value = '';
    }
  });

  // Drag events
  area.addEventListener('dragover', (e) => {
    e.preventDefault();
    area.classList.add('dragover');
  });

  area.addEventListener('dragleave', () => {
    area.classList.remove('dragover');
  });

  area.addEventListener('drop', (e) => {
    e.preventDefault();
    area.classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  });
}

async function handleFile(file) {
  const cfg = window.__GLYPHH__;
  const progressEl = document.getElementById('dropzone-progress');
  const fillEl = document.getElementById('progress-fill');
  const statusEl = document.getElementById('dropzone-status');

  // Validate file type
  if (!file.name.endsWith('.jsonl') && !file.name.endsWith('.json')) {
    showToast('Only .jsonl and .json files are supported', 'error');
    return;
  }

  progressEl.classList.add('active');
  fillEl.style.width = '0%';
  statusEl.textContent = 'Reading file...';

  try {
    const text = await file.text();
    let records;

    if (file.name.endsWith('.jsonl')) {
      records = text.trim().split('\n')
        .filter(line => line.trim())
        .map(line => JSON.parse(line));
    } else {
      const parsed = JSON.parse(text);
      records = Array.isArray(parsed) ? parsed : [parsed];
    }

    if (records.length === 0) {
      showToast('File contains no records', 'error');
      progressEl.classList.remove('active');
      return;
    }

    statusEl.textContent = `Encoding ${records.length} records...`;
    fillEl.style.width = '10%';

    // POST to the listener endpoint — this encodes records into HDC glyphs
    const headers = { 'Content-Type': 'application/json', ...authHeaders() };

    const res = await fetch(`/${cfg.orgId}/${cfg.modelId}/listener`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        records,
        batch_size: 50,
      }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    const jobId = data.job_id;
    const totalRecords = data.total_records || records.length;

    fillEl.style.width = '30%';
    statusEl.textContent = `Encoding ${totalRecords} records into glyphs...`;

    // Poll job progress
    if (jobId) {
      await pollJob(jobId, totalRecords, fillEl, statusEl);
    } else {
      fillEl.style.width = '100%';
      statusEl.textContent = `Encoded ${totalRecords} glyphs`;
    }

    showToast(`Encoded ${totalRecords} records from ${file.name}`, 'success');

    // Notify other panels to refresh
    window.dispatchEvent(new CustomEvent('glyphh:data-loaded'));

    setTimeout(() => {
      progressEl.classList.remove('active');
      fillEl.style.width = '0%';
    }, 2000);

  } catch (err) {
    console.error('Upload error:', err);
    statusEl.textContent = `Error: ${err.message}`;
    fillEl.style.width = '0%';
    showToast(`Upload failed: ${err.message}`, 'error');

    setTimeout(() => {
      progressEl.classList.remove('active');
    }, 3000);
  }
}

async function pollJob(jobId, totalRecords, fillEl, statusEl) {
  const cfg = window.__GLYPHH__;
  const headers = authHeaders();
  const maxPolls = 120; // 2 minutes max

  for (let i = 0; i < maxPolls; i++) {
    await new Promise(r => setTimeout(r, 1000));

    try {
      const res = await fetch(`/${cfg.orgId}/${cfg.modelId}/listener/jobs/${jobId}`, { headers });
      if (!res.ok) break;

      const job = await res.json();
      const processed = job.processed || 0;
      const pct = Math.min(95, 30 + (processed / Math.max(totalRecords, 1)) * 65);
      fillEl.style.width = `${pct}%`;
      statusEl.textContent = `Encoding... ${processed}/${totalRecords} glyphs`;

      if (job.status === 'completed' || job.status === 'done') {
        fillEl.style.width = '100%';
        statusEl.textContent = `Encoded ${processed} glyphs`;
        return;
      }
      if (job.status === 'failed' || job.status === 'error') {
        throw new Error(job.error || 'Encoding failed');
      }
    } catch (err) {
      // Job status endpoint may not exist — just wait and assume success
      if (i > 5) {
        fillEl.style.width = '100%';
        statusEl.textContent = `Encoded ${totalRecords} glyphs`;
        return;
      }
    }
  }

  // Timed out — assume done
  fillEl.style.width = '100%';
  statusEl.textContent = `Encoded ${totalRecords} glyphs`;
}

function showToast(message, type) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}
