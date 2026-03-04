/**
 * Query results panel — renders FactTree results from chat/GQL queries.
 *
 * Listens for 'glyphh:result' events dispatched by chat.js.
 */

let resultCount = 0;

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function getMatchNodes(ft) {
  if (!ft || !ft.children) return [];
  for (const child of ft.children) {
    if (child.description === 'results' && Array.isArray(child.children)) {
      return child.children.filter(c => c.description && c.description.startsWith('Match'));
    }
  }
  return [];
}

function resultLabel(ft) {
  if (!ft) return 'empty';
  if (ft.description === 'Query Error') return 'error';
  const matches = getMatchNodes(ft);
  if (matches.length > 0) return matches.length + ' result' + (matches.length !== 1 ? 's' : '');
  const rNode = (ft?.children || []).find(c => c.description === 'results');
  if (rNode?.children?.length) return rNode.children.length + ' glyph' + (rNode.children.length !== 1 ? 's' : '');
  return 'view';
}

export function init() {
  const clearBtn = document.getElementById('clear-results-btn');
  clearBtn.addEventListener('click', clearResults);

  // Listen for results from chat
  window.addEventListener('glyphh:result', (e) => {
    addResult(e.detail);
  });
}

function addResult({ query, ft, tab, state }) {
  const body = document.getElementById('results-body');
  const empty = document.getElementById('results-empty');
  if (empty) empty.style.display = 'none';

  resultCount++;

  const item = document.createElement('div');
  item.className = 'result-item';

  const count = resultLabel(ft);
  const jsonStr = JSON.stringify(ft, null, 2);
  const stateClass = state ? `state-badge state-${state}` : '';
  const stateBadge = state ? `<span class="${stateClass}">${escHtml(state)}</span>` : '';

  item.innerHTML = `
    <details>
      <summary>
        <span class="summary-arrow">\u25B6</span>
        ${stateBadge}
        <span class="summary-label">${escHtml(query.substring(0, 50))}${query.length > 50 ? '\u2026' : ''}</span>
        <span class="summary-count">${count}</span>
      </summary>
      <div class="result-query-label">${escHtml(tab.toUpperCase())} \u00B7 #${resultCount}</div>
      <pre class="result-pre">${escHtml(jsonStr)}</pre>
    </details>`;

  body.insertBefore(item, body.firstChild);
}

function clearResults() {
  const body = document.getElementById('results-body');
  const empty = document.getElementById('results-empty');
  body.querySelectorAll('.result-item').forEach(el => el.remove());
  resultCount = 0;
  if (empty) empty.style.display = 'flex';
}
