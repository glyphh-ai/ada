/**
 * Glyph list panel — searchable, expandable list of glyphs.
 *
 * Fetches from GET /{org_id}/{model_id}/data
 * Each glyph is a collapsible <details> showing full JSON on expand.
 * Click glyph → dispatches 'glyphh:glyph-select' for viewer highlight.
 */

import { authHeaders } from './auth.js';

let allGlyphs = [];
let filteredGlyphs = [];
let selectedId = null;
let totalCount = 0;

const TYPE_COLORS = {
  model: '#1d4ed8',
  encoder: '#7c3aed',
  listener: '#38bdf8',
  profile: '#f97316',
  map: '#4ade80',
  trend: '#14b8a6',
  layer: '#a855f7',
  segment: '#f472b6',
  role: '#0ea5e9',
  default: '#64748b',
};

function getTypeColor(nodeType) {
  if (!nodeType) return TYPE_COLORS.default;
  return TYPE_COLORS[String(nodeType).toLowerCase()] || TYPE_COLORS.default;
}

export function init() {
  const searchInput = document.getElementById('glyph-search-input');
  const refreshBtn = document.getElementById('refresh-glyphs-btn');

  searchInput.addEventListener('input', () => {
    filterGlyphs(searchInput.value.trim().toLowerCase());
  });

  refreshBtn.addEventListener('click', () => loadGlyphs());
  window.addEventListener('glyphh:data-loaded', () => loadGlyphs());

  loadGlyphs();
}

export async function loadGlyphs() {
  const cfg = window.__GLYPHH__;
  const headers = authHeaders();

  try {
    const countRes = await fetch(`/${cfg.orgId}/${cfg.modelId}/data/count`, { headers });
    if (countRes.ok) {
      const countData = await countRes.json();
      totalCount = countData.glyphs || 0;
      updateCountBadge();
    }

    const res = await fetch(`/${cfg.orgId}/${cfg.modelId}/data?limit=500&offset=0`, { headers });
    if (!res.ok) {
      allGlyphs = [];
      filteredGlyphs = [];
      render();
      return;
    }

    const data = await res.json();
    allGlyphs = data.glyphs || [];
    totalCount = data.total || allGlyphs.length;
    updateCountBadge();

    const searchInput = document.getElementById('glyph-search-input');
    const q = searchInput ? searchInput.value.trim().toLowerCase() : '';
    filterGlyphs(q);
  } catch (err) {
    console.error('Failed to load glyphs:', err);
    allGlyphs = [];
    filteredGlyphs = [];
    render();
  }
}

function updateCountBadge() {
  const badge = document.getElementById('glyph-count-badge');
  if (badge) {
    badge.textContent = totalCount > 0 ? `${totalCount} glyphs` : '';
  }
}

function filterGlyphs(query) {
  if (!query) {
    filteredGlyphs = allGlyphs;
  } else {
    filteredGlyphs = allGlyphs.filter(g => {
      const text = (g.concept_text || '').toLowerCase();
      const meta = JSON.stringify(g.metadata || {}).toLowerCase();
      return text.includes(query) || meta.includes(query);
    });
  }
  render();
}

function render() {
  const list = document.getElementById('glyph-list');
  const empty = document.getElementById('glyph-empty');

  // Clear old items
  list.querySelectorAll('.viewer-item, .glyph-count').forEach(el => el.remove());

  if (filteredGlyphs.length === 0) {
    empty.style.display = 'flex';
    return;
  }

  empty.style.display = 'none';

  // Count label
  const countText = totalCount > filteredGlyphs.length
    ? `Showing ${filteredGlyphs.length} of ${totalCount}`
    : `${filteredGlyphs.length} glyph${filteredGlyphs.length !== 1 ? 's' : ''}`;
  list.insertAdjacentHTML('beforeend', `<div class="glyph-count">${countText}</div>`);

  // Render each glyph as an expandable item
  for (const g of filteredGlyphs) {
    const name = g.concept_text || g.id || 'unnamed';
    const nodeType = g.metadata?.node_type || '';
    const color = getTypeColor(nodeType);
    const jsonStr = JSON.stringify(g, null, 2);
    const propCount = Object.keys(g.metadata || {}).length;

    const badge = nodeType
      ? `<span class="glyph-type-badge" style="background:${color}22;color:${color};border:1px solid ${color}44">${escHtml(nodeType)}</span>`
      : '';

    const item = document.createElement('div');
    item.className = 'viewer-item';
    item.dataset.glyphId = g.id || '';

    item.innerHTML = `
      <details>
        <summary>
          <span class="summary-arrow">\u25B6</span>
          ${badge}
          <span class="summary-label">${escHtml(name)}</span>
          <span class="summary-count">${propCount} prop${propCount !== 1 ? 's' : ''}</span>
        </summary>
        <pre class="result-pre">${escHtml(jsonStr)}</pre>
      </details>`;

    // Click summary to select (dispatch event for 3D viewer)
    const summary = item.querySelector('summary');
    summary.addEventListener('click', (e) => {
      // Toggle selection on single click
      const id = g.id;
      if (selectedId === id) {
        selectedId = null;
        window.dispatchEvent(new CustomEvent('glyphh:glyph-select', { detail: null }));
      } else {
        selectedId = id;
        window.dispatchEvent(new CustomEvent('glyphh:glyph-select', { detail: g }));
      }
      // Update highlight styles
      list.querySelectorAll('.viewer-item').forEach(el => {
        el.classList.toggle('highlighted', el.dataset.glyphId === selectedId);
      });
    });

    list.appendChild(item);
  }
}

function escHtml(s) {
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
