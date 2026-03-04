/**
 * Data upload dropzone — drag-and-drop .jsonl files.
 *
 * Reads file, POSTs each line as a glyph concept to the MCP load endpoint.
 * Dispatches 'glyphh:data-loaded' on success.
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
    let concepts;

    if (file.name.endsWith('.jsonl')) {
      // JSONL: one JSON object per line
      concepts = text.trim().split('\n')
        .filter(line => line.trim())
        .map(line => JSON.parse(line));
    } else {
      // JSON: expect an array
      const parsed = JSON.parse(text);
      concepts = Array.isArray(parsed) ? parsed : [parsed];
    }

    statusEl.textContent = `Uploading ${concepts.length} glyphs...`;
    fillEl.style.width = '20%';

    // Use MCP load_data tool to upload in batch
    const headers = { 'Content-Type': 'application/json', ...authHeaders() };

    const res = await fetch(`/${cfg.orgId}/${cfg.modelId}/mcp`, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        tool: 'load_data',
        arguments: {
          concepts,
          org_id: cfg.orgId,
          model_id: cfg.modelId,
        },
      }),
    });

    fillEl.style.width = '80%';

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || err.error || `HTTP ${res.status}`);
    }

    const data = await res.json();
    fillEl.style.width = '100%';

    const loaded = data.content?.[0]?.data?.loaded || concepts.length;
    statusEl.textContent = `Loaded ${loaded} glyphs`;

    showToast(`Loaded ${loaded} glyphs from ${file.name}`, 'success');

    // Notify other panels
    window.dispatchEvent(new CustomEvent('glyphh:data-loaded'));

    // Reset after a moment
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

function showToast(message, type) {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.textContent = message;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3000);
}
