/**
 * Chat / GQL panel — ported from org_scoped.py inline HTML.
 *
 * NL tab + GQL tab with tab switching.
 * Calls /{org_id}/{model_id}/mcp endpoint.
 * Dispatches 'glyphh:result' custom events for the results panel.
 */

import { authHeaders } from './auth.js';

let currentTab = 'nl';
let busy = false;

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

export function init() {
  const cfg = window.__GLYPHH__;
  const tabNl = document.getElementById('tab-nl');
  const tabGql = document.getElementById('tab-gql');
  const nlInput = document.getElementById('nl-input');
  const gqlInput = document.getElementById('gql-input');
  const nlBtn = document.getElementById('nl-btn');
  const gqlBtn = document.getElementById('gql-btn');

  tabNl.addEventListener('click', () => switchTab('nl'));
  tabGql.addEventListener('click', () => switchTab('gql'));

  nlInput.addEventListener('input', syncBtn);
  gqlInput.addEventListener('input', syncBtn);

  nlInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); }
  });
  gqlInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && e.ctrlKey) { e.preventDefault(); submit(); }
  });

  nlBtn.addEventListener('click', submit);
  gqlBtn.addEventListener('click', submit);

  nlInput.focus();
}

function switchTab(tab) {
  currentTab = tab;
  document.getElementById('tab-nl').classList.toggle('active', tab === 'nl');
  document.getElementById('tab-gql').classList.toggle('active', tab === 'gql');
  document.getElementById('nl-wrap').classList.toggle('hidden', tab !== 'nl');
  document.getElementById('gql-wrap').classList.toggle('hidden', tab !== 'gql');
  document.getElementById('input-hint').textContent = tab === 'nl' ? 'Press Enter to send' : 'Press Ctrl+Enter to send';
  (tab === 'nl' ? document.getElementById('nl-input') : document.getElementById('gql-input')).focus();
  syncBtn();
}

function getInput() {
  const el = document.getElementById(currentTab === 'nl' ? 'nl-input' : 'gql-input');
  return el ? el.value.trim() : '';
}

function clearInput() {
  const el = document.getElementById(currentTab === 'nl' ? 'nl-input' : 'gql-input');
  if (el) el.value = '';
  syncBtn();
}

function syncBtn() {
  const hasText = getInput().length > 0;
  const nlBtn = document.getElementById('nl-btn');
  const gqlBtn = document.getElementById('gql-btn');
  if (nlBtn) nlBtn.disabled = !hasText || busy;
  if (gqlBtn) gqlBtn.disabled = !hasText || busy;
}

function addMessage(role, html, meta) {
  const msgs = document.getElementById('chat-messages');
  const row = document.createElement('div');
  row.className = 'msg-row ' + role;
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = html;
  row.appendChild(bubble);
  if (meta) {
    const m = document.createElement('div');
    m.className = 'msg-meta';
    m.innerHTML = meta;
    row.appendChild(m);
  }
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;
  return row;
}

function addThinking() {
  const msgs = document.getElementById('chat-messages');
  const row = document.createElement('div');
  row.className = 'msg-row assistant';
  row.innerHTML = '<div class="thinking"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>';
  msgs.appendChild(row);
  msgs.scrollTop = msgs.scrollHeight;
  return row;
}

function removeEl(el) {
  if (el && el.parentNode) el.parentNode.removeChild(el);
}

// Walk the FactTree to find Match nodes
function getMatchNodes(ft) {
  if (!ft || !ft.children) return [];
  for (const child of ft.children) {
    if (child.description === 'results' && Array.isArray(child.children)) {
      return child.children.filter(c => c.description && c.description.startsWith('Match'));
    }
  }
  return [];
}

function summarizeResult(ft) {
  if (!ft) return 'No result';

  if (ft.description === 'Query Error') {
    const errChild = (ft.children || []).find(c => c.description === 'Error Details');
    return '\u26A0 ' + escHtml(String(errChild?.value || 'Query failed'));
  }

  const matches = getMatchNodes(ft);
  if (matches.length > 0) {
    return matches.map(m => {
      const score = m.value?.final_score ?? 0;
      const pct = (score * 100).toFixed(1);
      const filled = Math.round(score * 12);
      const bar = '\u2588'.repeat(filled) + '\u2591'.repeat(12 - filled);
      const name = escHtml(m.value?.concept_text || 'match');
      return `<div class="match-row"><span class="match-pct">${pct}%</span><span class="match-bar">[${bar}]</span><span class="match-name">${name}</span></div>`;
    }).join('');
  }

  if (ft.description === 'Similarity Search') return 'No matches found';
  if (ft.description === 'List Query') {
    const rNode = (ft.children || []).find(c => c.description === 'results');
    const n = rNode?.children?.length || 0;
    return n + ' glyph' + (n !== 1 ? 's' : '');
  }
  if (ft.description) return escHtml(ft.description);
  return 'Result received';
}

async function submit() {
  const cfg = window.__GLYPHH__;
  const query = getInput();
  if (!query || busy) return;
  busy = true;
  syncBtn();

  addMessage('user', escHtml(query), null);
  clearInput();
  const thinking = addThinking();

  try {
    const tool = currentTab === 'nl' ? 'nl_query' : 'gql_query';
    const args = currentTab === 'nl'
      ? { query }
      : { query, enable_cache: true };

    const headers = { 'Content-Type': 'application/json', ...authHeaders() };

    const res = await fetch(`/${cfg.orgId}/${cfg.modelId}/mcp`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ tool, arguments: args }),
    });

    const data = await res.json();
    removeEl(thinking);

    const ft = data.result;
    const isQueryError = ft?.description === 'Query Error';
    const state = data.content?.[0]?.data?.state || (ft ? 'DONE' : null);

    if (data.isError || !res.ok) {
      const errMsg = data.error || data.detail || 'Unknown error';
      addMessage('error', escHtml(errMsg), null);
    } else if (isQueryError) {
      const errChild = (ft?.children || []).find(c => c.description === 'Error Details');
      addMessage('error', escHtml(String(errChild?.value || 'Query error')), null);
    } else {
      const ms = typeof data.query_time_ms === 'number' ? data.query_time_ms.toFixed(1) + 'ms' : null;
      const method = data.match_method;
      const qtype = data.query_type;

      let metaParts = [];
      if (state) metaParts.push(`<span class="pill state-badge state-${state}">${escHtml(state)}</span>`);
      if (ms) metaParts.push(`<span>${ms}</span>`);
      if (method && method !== 'none') metaParts.push(`<span class="pill">${escHtml(method)}</span>`);
      if (qtype && qtype !== 'unknown') metaParts.push(`<span class="pill">${escHtml(qtype)}</span>`);

      if (state === 'ASK') {
        const askData = data.content?.[0]?.data?.ask || {};
        const question = askData.question || 'Please clarify your request.';
        let html = escHtml(question);
        const options = askData.disambiguation_options || [];
        if (options.length > 0) {
          html += '<ul style="margin:6px 0 0;padding-left:20px;list-style:none;">';
          for (const opt of options) {
            const label = opt.suggestion || opt.intent || String(opt);
            html += `<li style="margin:2px 0;">\u2022 ${escHtml(label)}</li>`;
          }
          html += '</ul>';
        }
        addMessage('assistant', html, metaParts.join(''));
      } else {
        const summary = summarizeResult(ft);
        addMessage('assistant', summary, metaParts.join(''));
      }

      // Dispatch result event for results panel and viewer
      window.dispatchEvent(new CustomEvent('glyphh:result', {
        detail: { query, ft, tab: currentTab, state, data },
      }));
    }
  } catch (err) {
    removeEl(thinking);
    addMessage('error', 'Network error: ' + escHtml(err.message), null);
  }

  busy = false;
  syncBtn();
  (currentTab === 'nl' ? document.getElementById('nl-input') : document.getElementById('gql-input')).focus();
}
