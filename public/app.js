/**
 * Glyphh Runtime Dashboard
 *
 * Vanilla JS SPA — no build step.
 * Screens: login, home (model list), detail (model data + drag-and-drop).
 */

const App = (() => {
  // ── State ──────────────────────────────────────────────────────────────
  let orgId = null;
  let token = null;
  let currentModelId = null;
  let currentModel = null;
  let dataOffset = 0;
  const DATA_LIMIT = 20;
  let _dialogResolve = null;
  let _pollTimer = null;

  // ── Helpers ────────────────────────────────────────────────────────────

  function api(method, path, body) {
    const opts = { method, headers: {} };
    if (token) opts.headers['Authorization'] = `Bearer ${token}`;
    if (body instanceof FormData) {
      opts.body = body;
    } else if (body) {
      opts.headers['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts).then(async res => {
      const data = await res.json().catch(() => null);
      if (!res.ok) throw new Error((data && (data.detail || data.error?.message)) || `HTTP ${res.status}`);
      return data;
    });
  }

  function showScreen(name) {
    document.querySelectorAll('[data-screen]').forEach(el => {
      el.hidden = el.dataset.screen !== name;
    });
  }

  function toast(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    document.getElementById('toast-container').appendChild(el);
    setTimeout(() => el.remove(), 4000);
  }

  function escHtml(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  // ── Init ───────────────────────────────────────────────────────────────

  async function init() {
    token = localStorage.getItem('glyphh_token');
    orgId = localStorage.getItem('glyphh_org_id');

    try {
      const session = await api('GET', '/ui/session');

      // Show version on login screen
      if (session.version) {
        const heading = document.getElementById('login-heading');
        if (heading) heading.textContent = `Glyphh Ada v${session.version}`;
      }

      // Check if we have a valid token
      if (session.authenticated && session.org_id) {
        orgId = session.org_id;
        localStorage.setItem('glyphh_org_id', orgId);
        showHome();
        return;
      }
    } catch (e) {
      // Session check failed — try local fallback
      console.warn('Session check failed:', e);
    }

    // Not authenticated — show login
    if (!token) {
      showScreen('login');
    } else {
      // Token exists but session invalid — clear and show login
      localStorage.removeItem('glyphh_token');
      localStorage.removeItem('glyphh_org_id');
      token = null;
      orgId = null;
      showScreen('login');
    }
  }

  // ── Auth (Device Flow) ─────────────────────────────────────────────────

  async function startLogin() {
    document.getElementById('login-prompt').hidden = true;
    document.getElementById('login-polling').hidden = false;
    document.getElementById('login-error').hidden = true;

    try {
      const data = await api('POST', '/ui/auth/device/start');
      document.getElementById('device-code').textContent = data.user_code;

      // Open Studio verification URL in new tab
      window.open(data.verification_url, '_blank');

      // Poll for approval
      _pollTimer = setInterval(async () => {
        try {
          const poll = await api('POST', '/ui/auth/device/poll', {
            device_code: data.device_code,
          });

          if (poll.status === 'approved') {
            clearInterval(_pollTimer);
            _pollTimer = null;

            token = poll.access_token;
            localStorage.setItem('glyphh_token', token);
            if (poll.refresh_token) {
              localStorage.setItem('glyphh_refresh_token', poll.refresh_token);
            }
            if (poll.user?.org_id) {
              orgId = poll.user.org_id;
              localStorage.setItem('glyphh_org_id', orgId);
            }

            toast(`Welcome, ${poll.user?.first_name || 'there'}!`, 'success');
            showHome();
          } else if (poll.status === 'expired') {
            clearInterval(_pollTimer);
            _pollTimer = null;
            showLoginError('Code expired. Please try again.');
          }
        } catch (e) {
          // Keep polling on network errors
          console.warn('Poll error:', e);
        }
      }, data.interval * 1000 || 5000);

    } catch (e) {
      showLoginError(e.message || 'Could not start login. Is the platform reachable?');
    }
  }

  function showLoginError(msg) {
    document.getElementById('login-prompt').hidden = true;
    document.getElementById('login-polling').hidden = true;
    document.getElementById('login-error').hidden = false;
    document.getElementById('login-error-msg').textContent = msg;
  }

  function resetLogin() {
    if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
    document.getElementById('login-prompt').hidden = false;
    document.getElementById('login-polling').hidden = true;
    document.getElementById('login-error').hidden = true;
  }

  function logout() {
    if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
    localStorage.removeItem('glyphh_token');
    localStorage.removeItem('glyphh_refresh_token');
    localStorage.removeItem('glyphh_org_id');
    token = null;
    orgId = null;
    resetLogin();
    showScreen('login');
  }

  // ── Home Screen ────────────────────────────────────────────────────────

  async function showHome() {
    showScreen('home');

    // Update header
    document.getElementById('header-org').textContent = orgId || '';
    document.getElementById('logout-btn').hidden = false;

    await loadModels();
  }

  async function loadModels() {
    const grid = document.getElementById('model-grid');
    const empty = document.getElementById('model-empty');

    try {
      const data = await api('GET', `/${orgId}/models`);
      const models = data.models || [];

      if (models.length === 0) {
        grid.innerHTML = '';
        empty.hidden = false;
        return;
      }

      empty.hidden = true;
      grid.innerHTML = models.map(m => `
        <div class="model-card" onclick="App.showDetail('${escHtml(m.model_id)}')">
          <div class="model-card-header">
            <span class="model-card-name">${escHtml(m.name || m.model_id)}</span>
            ${m.version ? `<span class="model-card-version">v${escHtml(m.version)}</span>` : ''}
          </div>
          <div class="model-card-id">${escHtml(m.model_id)}</div>
          <div class="model-card-stats">
            <span class="model-card-stat">
              <span class="status-dot ${m.status === 'active' ? 'active' : 'inactive'}"></span>
              ${escHtml(m.status || 'unknown')}
            </span>
            <span class="model-card-stat"><b>${m.glyphs ?? 0}</b> glyphs</span>
          </div>
        </div>
      `).join('');

    } catch (e) {
      grid.innerHTML = '';
      empty.hidden = false;
      if (e.message && !e.message.includes('404')) {
        toast('Failed to load models: ' + e.message, 'error');
      }
    }
  }

  // ── Deploy ─────────────────────────────────────────────────────────────

  async function handleDeploy(input) {
    const file = input.files[0];
    if (!file) return;
    input.value = '';

    const modelId = file.name.replace(/\.glyphh$/, '');
    toast(`Deploying ${modelId}...`, 'info');

    try {
      const form = new FormData();
      form.append('file', file, file.name);
      await api('POST', `/${orgId}/${modelId}/model/deploy`, form);
      toast(`Deployed: ${modelId}`, 'success');
      await loadModels();
    } catch (e) {
      toast('Deploy failed: ' + e.message, 'error');
    }
  }

  // ── Model Detail ───────────────────────────────────────────────────────

  async function showDetail(modelId) {
    currentModelId = modelId;
    dataOffset = 0;
    showScreen('detail');

    document.getElementById('detail-header-org').textContent = orgId || '';
    document.getElementById('detail-name').textContent = modelId;
    document.getElementById('detail-id').textContent = modelId;
    document.getElementById('detail-version').textContent = '';
    document.getElementById('stat-glyphs').textContent = '-';
    document.getElementById('stat-vectors').textContent = '-';
    document.getElementById('stat-status').textContent = '-';
    document.getElementById('load-progress').hidden = true;

    // Fetch status + counts in parallel
    try {
      const [ready, counts] = await Promise.all([
        api('GET', `/${orgId}/${modelId}/ready`).catch(() => null),
        api('GET', `/${orgId}/${modelId}/data/count`).catch(() => null),
      ]);

      if (ready) {
        document.getElementById('detail-name').textContent = ready.meta_name || modelId;
        document.getElementById('stat-status').textContent = ready.ready ? 'Ready' : (ready.status || 'Unknown');
        document.getElementById('stat-status').style.color = ready.ready ? 'var(--g-success)' : 'var(--g-warning)';
      }

      if (counts) {
        document.getElementById('stat-glyphs').textContent = (counts.glyphs ?? 0).toLocaleString();
        document.getElementById('stat-vectors').textContent = (counts.vectors ?? 0).toLocaleString();
      }
    } catch (e) {
      console.warn('Detail fetch error:', e);
    }

    await loadDataTable();
  }

  async function loadDataTable() {
    const tbody = document.getElementById('data-table-body');
    const info = document.getElementById('data-table-info');
    const pagination = document.getElementById('data-pagination');

    try {
      const data = await api('GET', `/${orgId}/${currentModelId}/data?limit=${DATA_LIMIT}&offset=${dataOffset}`);
      const glyphs = data.glyphs || [];
      const total = data.total || 0;

      if (glyphs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="3" style="text-align:center;color:var(--g-muted);padding:24px">No data loaded yet</td></tr>';
        info.textContent = '0 records';
        pagination.innerHTML = '';
        return;
      }

      tbody.innerHTML = glyphs.map(g => {
        const meta = g.metadata || {};
        const metaStr = Object.entries(meta).slice(0, 3).map(([k, v]) =>
          `${escHtml(k)}=${escHtml(String(v).slice(0, 30))}`
        ).join(', ');
        return `
          <tr>
            <td style="font-family:var(--g-mono);font-size:11px;color:var(--g-muted)">${escHtml(String(g.id).slice(0, 8))}</td>
            <td>${escHtml((g.concept_text || '').slice(0, 80))}</td>
            <td style="font-size:12px;color:var(--g-muted)">${escHtml(metaStr)}</td>
          </tr>
        `;
      }).join('');

      const from = dataOffset + 1;
      const to = dataOffset + glyphs.length;
      info.textContent = `${from}-${to} of ${total}`;

      // Pagination buttons
      let btns = '';
      if (dataOffset > 0) {
        btns += `<button class="btn btn-sm" onclick="App.prevPage()">Prev</button>`;
      }
      if (dataOffset + DATA_LIMIT < total) {
        btns += `<button class="btn btn-sm" onclick="App.nextPage()">Next</button>`;
      }
      pagination.innerHTML = btns;

    } catch (e) {
      tbody.innerHTML = '<tr><td colspan="3" style="color:var(--g-muted);padding:24px">Could not load data</td></tr>';
      info.textContent = '';
      pagination.innerHTML = '';
    }
  }

  function nextPage() { dataOffset += DATA_LIMIT; loadDataTable(); }
  function prevPage() { dataOffset = Math.max(0, dataOffset - DATA_LIMIT); loadDataTable(); }

  // ── Model Actions ──────────────────────────────────────────────────────

  async function reEncodeModel() {
    try {
      await api('POST', `/${orgId}/${currentModelId}/model/re-encode`);
      toast('Re-encode started', 'success');
    } catch (e) {
      toast('Re-encode failed: ' + e.message, 'error');
    }
  }

  async function confirmClear() {
    const ok = await showDialog(
      'Clear Data',
      `This will delete all glyphs and edges for ${currentModelId}. The model stays loaded.`
    );
    if (!ok) return;

    try {
      const data = await api('DELETE', `/${orgId}/${currentModelId}/data`);
      toast(`Cleared: ${data.glyphs_deleted ?? 0} glyphs`, 'success');
      await showDetail(currentModelId);
    } catch (e) {
      toast('Clear failed: ' + e.message, 'error');
    }
  }

  async function confirmDelete() {
    const ok = await showDialog(
      'Delete Model',
      `This will permanently remove ${currentModelId} and all its data from the runtime.`
    );
    if (!ok) return;

    try {
      await api('DELETE', `/${orgId}/${currentModelId}/model`);
      toast(`Deleted: ${currentModelId}`, 'success');
      showHome();
    } catch (e) {
      toast('Delete failed: ' + e.message, 'error');
    }
  }

  // ── Confirm Dialog ─────────────────────────────────────────────────────

  function showDialog(title, message) {
    document.getElementById('dialog-title').textContent = title;
    document.getElementById('dialog-message').textContent = message;
    document.getElementById('dialog-overlay').hidden = false;
    return new Promise(resolve => { _dialogResolve = resolve; });
  }

  function dialogConfirm() {
    document.getElementById('dialog-overlay').hidden = true;
    if (_dialogResolve) { _dialogResolve(true); _dialogResolve = null; }
  }

  function dialogCancel() {
    document.getElementById('dialog-overlay').hidden = true;
    if (_dialogResolve) { _dialogResolve(false); _dialogResolve = null; }
  }

  // ── Drag and Drop Data Loading ─────────────────────────────────────────

  function dragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('drag-over');
  }

  function dragLeave(e) {
    e.currentTarget.classList.remove('drag-over');
  }

  function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) loadDataFile(file);
  }

  function handleDataFile(input) {
    const file = input.files[0];
    input.value = '';
    if (file) loadDataFile(file);
  }

  async function loadDataFile(file) {
    const progressEl = document.getElementById('load-progress');
    const fillEl = document.getElementById('progress-fill');
    const statusEl = document.getElementById('load-status');

    // Read and parse the file
    let records;
    try {
      const text = await file.text();
      records = parseRecords(text);
    } catch (e) {
      toast('Failed to parse file: ' + e.message, 'error');
      return;
    }

    if (!records || records.length === 0) {
      toast('No records found in file', 'error');
      return;
    }

    // Show progress
    progressEl.hidden = false;
    fillEl.style.width = '0%';
    statusEl.textContent = `Loading ${records.length} records...`;

    try {
      const data = await api('POST', `/${orgId}/${currentModelId}/listener`, {
        records: records,
        batch_size: 50,
      });

      const jobId = data.job_id;
      if (!jobId) {
        fillEl.style.width = '100%';
        statusEl.textContent = `Loaded ${records.length} records`;
        toast('Data loaded successfully', 'success');
        await showDetail(currentModelId);
        return;
      }

      // Poll job progress
      const pollJob = setInterval(async () => {
        try {
          const job = await api('GET', `/${orgId}/${currentModelId}/listener/jobs/${jobId}`);
          const total = job.total_records || records.length;
          const processed = job.processed_records || 0;
          const pct = total > 0 ? Math.round((processed / total) * 100) : 0;

          fillEl.style.width = pct + '%';
          statusEl.textContent = `${processed} / ${total} records (${job.status || 'processing'})`;

          if (job.status === 'completed' || job.status === 'done') {
            clearInterval(pollJob);
            fillEl.style.width = '100%';
            statusEl.textContent = `Loaded ${total} records`;
            toast('Data loaded successfully', 'success');
            await showDetail(currentModelId);
          } else if (job.status === 'error' || job.status === 'failed') {
            clearInterval(pollJob);
            statusEl.textContent = `Error: ${job.error || 'Unknown error'}`;
            toast('Data load failed', 'error');
          }
        } catch (e) {
          clearInterval(pollJob);
          statusEl.textContent = 'Error checking progress';
          toast('Lost connection to job', 'error');
        }
      }, 2000);

    } catch (e) {
      progressEl.hidden = true;
      toast('Data load failed: ' + e.message, 'error');
    }
  }

  function parseRecords(text) {
    text = text.trim();
    // Try JSON first (array or {records: [...]})
    try {
      const data = JSON.parse(text);
      if (Array.isArray(data)) return data;
      if (data.records && Array.isArray(data.records)) return data.records;
    } catch (_) {}

    // Fall back to JSONL
    const records = [];
    for (const line of text.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      records.push(JSON.parse(trimmed));
    }
    return records;
  }

  // ── Public API ─────────────────────────────────────────────────────────

  return {
    init,
    startLogin,
    resetLogin,
    logout,
    showHome,
    showDetail,
    handleDeploy,
    loadModels,
    reEncodeModel,
    confirmClear,
    confirmDelete,
    dialogConfirm,
    dialogCancel,
    dragOver,
    dragLeave,
    handleDrop,
    handleDataFile,
    nextPage,
    prevPage,
  };
})();

// Boot
document.addEventListener('DOMContentLoaded', App.init);
