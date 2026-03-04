/**
 * 3D Glyph Viewer — faithful vanilla JS port of GlyphViewer.tsx from glyphh-studio-old.
 *
 * Features:
 * - Cortex point cloud rendering (boolean bit vectors → 3D point cloud)
 * - kNN similarity layout (Jaccard on cortex bits, rest-length springs)
 * - 3 edge types: semantic (orange), neural (blue), hierarchy (purple)
 * - Edge visibility toggles (checkboxes)
 * - Layer switching (keys 1-4)
 * - Explode mode (E), Freeze mode (F), Labels toggle (L)
 * - Highlight mode: click node → show connected neighbors, dim rest
 * - Search panel with prefix match
 * - HTML labels with 3D→2D projection
 * - Legend overlay with node type colors
 */

import { authHeaders } from './auth.js';

// ── Layout Constants (from original GlyphViewer.tsx) ──

const LAYOUT_PARAMS = {
  springSemantic: 0.06,
  springNeural: 0.05,
  springHierarchy: 0.07,
  repulsion: 14.0,
  damping: 0.9,
  maxStep: 0.55,
};

const CORTEX_LAYOUT = {
  enabled: true,
  K: 10,
  springK: 0.12,
  dMin: 2.5,
  dMaxScale: 1.2,
  repulsionBoost: 1.25,
  easePower: 2,
  layoutBlend: 1.0,
};

const EXPLODE_SCALE = 2.4;
const OVERLAY_TEXT = 'Glyph AI 3D Viewer \u2013 drag to rotate, scroll to zoom';

const TYPE_COLORS = {
  model: 0x1d4ed8, encoder: 0x7c3aed, listener: 0x38bdf8,
  profile: 0xf97316, map: 0x4ade80, trend: 0x14b8a6,
  layer: 0xa855f7, segment: 0xf472b6, role: 0x0ea5e9,
  taxonomy: 0x39ff14, concept: 0x00ffea, default: 0x64748b,
};

const TYPE_COLORS_CSS = {
  model: '#1d4ed8', encoder: '#7c3aed', listener: '#38bdf8',
  profile: '#f97316', map: '#4ade80', trend: '#14b8a6',
  layer: '#a855f7', segment: '#f472b6', role: '#0ea5e9',
  taxonomy: '#39ff14', concept: '#00ffea',
};

function computeLayoutRadius(n) {
  if (n < 20) return 22;
  if (n < 50) return 32;
  if (n < 100) return 42;
  if (n < 200) return 52;
  return 64 + Math.log(n) * 10;
}

// ── Module-level state ──

let THREE = null;
let OrbitControls = null;

// Three.js scene
let scene, camera, renderer, controls;
let animationId = null;

// Data
let glyphData = [];
let rolesConfig = null;
let edgesSemantic = [];
let edgesNeural = [];
let edgesHierarchy = [];
let edgesCortex = [];
const cortexBitsByName = new Map();

// Scene objects
let glyphGroups = [];
let glyphByName = new Map();
let semanticLines = null;
let neuralLines = null;
let hierarchyLines = null;
let brightLinesSemantic = null;
let brightLinesNeural = null;
let brightLinesHierarchy = null;

const nodeTypeByName = new Map();
const nodePositions = new Map();
const nodeVelocities = new Map();

// Viewer state
let activeLayer = 0;
let explodeMode = false;
let showSemantic = true;
let showNeural = true;
let showHierarchy = true;
let showLabels = true;
let highlightShowSemantic = true;
let highlightShowNeural = true;
let highlightShowHierarchy = true;
let selectedName = null;
let highlightMode = false;
let highlightSet = new Set();
let frozen = false;

// DOM refs
let containerEl, mountEl, emptyEl, labelsEl;
let overlayEl, layerKeyEl, legendEl, edgeTogglesEl, searchPanelEl;

let _worldPos = null; // reused Vector3

// ── Public API ──

export function init() {
  containerEl = document.getElementById('viewer-container');
  emptyEl = document.getElementById('viewer-empty');
  labelsEl = document.getElementById('viewer-labels');

  window.addEventListener('glyphh:data-loaded', () => loadViewer());
  window.addEventListener('glyphh:glyph-select', onGlyphSelect);
  window.addEventListener('glyphh:result', onResultHighlight);

  loadViewer();
}

// ── Cortex Similarity Helpers ──

function getLayerCortexBits(g, layerIndex) {
  const layer = g.layers?.find(l => l.index === layerIndex) ?? g.layers?.[0];
  return layer?.cortex ?? [];
}

function jaccardBits(a, b) {
  const n = Math.min(a.length, b.length);
  let inter = 0, uni = 0;
  for (let i = 0; i < n; i++) {
    const ai = !!a[i], bi = !!b[i];
    if (ai || bi) uni++;
    if (ai && bi) inter++;
  }
  return uni === 0 ? 0 : inter / uni;
}

function distanceFromSim(sim, dMin, dMax) {
  const t = 1 - Math.max(0, Math.min(1, sim));
  const p = Math.max(1, CORTEX_LAYOUT.easePower);
  const eased = Math.pow(t, p);
  return dMin + eased * (dMax - dMin);
}

function buildCortexEdgesKNN(K) {
  edgesCortex = [];
  const names = glyphData.map(g => g.name);
  const R = computeLayoutRadius(glyphData.length);
  const dMax = Math.max(CORTEX_LAYOUT.dMin + 1, R * CORTEX_LAYOUT.dMaxScale);

  for (let i = 0; i < names.length; i++) {
    const aName = names[i];
    const aBits = cortexBitsByName.get(aName) ?? [];
    const scored = [];

    for (let j = 0; j < names.length; j++) {
      if (i === j) continue;
      const bName = names[j];
      const bBits = cortexBitsByName.get(bName) ?? [];
      const sim = jaccardBits(aBits, bBits);
      if (sim > 0) scored.push({ name: bName, sim });
    }

    scored.sort((x, y) => y.sim - x.sim);
    const top = scored.slice(0, K);

    for (const t of top) {
      const rest = distanceFromSim(t.sim, CORTEX_LAYOUT.dMin, dMax);
      const weight = 0.15 + t.sim * 0.85;
      edgesCortex.push({ source: aName, target: t.name, sim: t.sim, rest, weight });
    }
  }

  // Dedup undirected edges, keep best sim
  const best = new Map();
  for (const e of edgesCortex) {
    const key = e.source < e.target ? `${e.source}|${e.target}` : `${e.target}|${e.source}`;
    const prev = best.get(key);
    if (!prev || e.sim > prev.sim) best.set(key, e);
  }
  edgesCortex = Array.from(best.values());
}

// ── Data Loading ──

async function loadViewer() {
  try {
    if (!THREE) {
      THREE = await import('three');
      const mod = await import('three/addons/controls/OrbitControls.js');
      OrbitControls = mod.OrbitControls;
      _worldPos = new THREE.Vector3();
    }

    cleanup();

    const cfg = window.__GLYPHH__;
    const headers = authHeaders();

    const res = await fetch(`/${cfg.orgId}/${cfg.modelId}/viewer/data?limit=200&ts=${Date.now()}`, {
      headers,
      cache: 'no-cache',
    });

    if (!res.ok) {
      console.error('Viewer data load failed:', res.status);
      showEmpty();
      return;
    }

    const json = await res.json();
    glyphData = json.glyphs || [];
    edgesSemantic = json.edges?.semantic || [];
    edgesNeural = json.edges?.neural || [];
    edgesHierarchy = json.edges?.hierarchy || [];
    rolesConfig = json.roles_config || null;

    if (!glyphData.length) {
      showEmpty();
      return;
    }

    // Build lookup maps
    nodeTypeByName.clear();
    glyphData.forEach(g => nodeTypeByName.set(g.name, g.node_type || 'concept'));

    cortexBitsByName.clear();
    glyphData.forEach(g => cortexBitsByName.set(g.name, getLayerCortexBits(g, activeLayer)));

    initThree();
    initLayout();

    if (CORTEX_LAYOUT.enabled) buildCortexEdgesKNN(CORTEX_LAYOUT.K);

    rebuildScene();
    buildUI();
    updateLayerKeyUI();

    // Show viewer
    emptyEl.style.display = 'none';

    animate();
    window.addEventListener('resize', onWindowResize);
    window.addEventListener('keydown', onKeyDown);
  } catch (err) {
    console.error('Viewer load error:', err);
  }
}

function showEmpty() {
  if (emptyEl) emptyEl.style.display = 'flex';
  if (mountEl) mountEl.innerHTML = '';
  if (labelsEl) labelsEl.innerHTML = '';
  removeUI();
}

// ── Three.js Init ──

function initThree() {
  const rect = containerEl.getBoundingClientRect();
  const width = rect.width || 800;
  const height = rect.height || 600;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x02040a);

  camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 2000);
  camera.position.set(0, 30, 70);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(window.devicePixelRatio);

  // Mount renderer canvas
  mountEl = document.getElementById('viewer-mount');
  if (!mountEl) {
    // Create mount div if not in HTML
    mountEl = document.createElement('div');
    mountEl.id = 'viewer-mount';
    mountEl.style.cssText = 'position:absolute;inset:0;';
    containerEl.appendChild(mountEl);
  }
  mountEl.innerHTML = '';
  mountEl.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.1;

  const light = new THREE.DirectionalLight(0xffffff, 1.0);
  light.position.set(10, 20, 10);
  scene.add(light);
}

function onWindowResize() {
  if (!camera || !renderer || !containerEl) return;
  const rect = containerEl.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return;
  camera.aspect = rect.width / rect.height;
  camera.updateProjectionMatrix();
  renderer.setSize(rect.width, rect.height);
}

// ── Layout ──

function initLayout() {
  nodePositions.clear();
  nodeVelocities.clear();
  if (!glyphData.length) return;

  // Initialize velocities
  glyphData.forEach(g => {
    if (!nodeVelocities.has(g.name)) nodeVelocities.set(g.name, new THREE.Vector3());
  });

  // Taxonomy ring near origin
  const taxonomyGlyphs = glyphData.filter(g => nodeTypeByName.get(g.name) === 'taxonomy');
  if (taxonomyGlyphs.length) {
    const baseRadius = computeLayoutRadius(glyphData.length);
    const taxRadius = Math.max(4, baseRadius * 0.35);
    taxonomyGlyphs.forEach((g, idx) => {
      const angle = (2 * Math.PI * idx) / taxonomyGlyphs.length;
      const heightOffset = idx % 2 === 0 ? 0.8 : 1.4;
      nodePositions.set(g.name, new THREE.Vector3(
        Math.cos(angle) * taxRadius, heightOffset, Math.sin(angle) * taxRadius
      ));
      nodeVelocities.set(g.name, new THREE.Vector3());
    });
  }

  // Semantic-type clustering for initial positions
  const typeSet = new Set(
    glyphData.map(g => g.semantic && (g.semantic.type || g.semantic.category)).filter(Boolean)
  );
  const types = Array.from(typeSet);
  const typeCenters = new Map();
  const R = computeLayoutRadius(glyphData.length);

  types.forEach((t, idx) => {
    const angle = (2 * Math.PI * idx) / Math.max(types.length, 1);
    typeCenters.set(t, new THREE.Vector3(Math.cos(angle) * R, 0, Math.sin(angle) * R));
  });

  glyphData.forEach(g => {
    if (nodeTypeByName.get(g.name) === 'taxonomy') return;
    const sem = g.semantic || {};
    const typeOrCat = sem.type || sem.category || 'unknown';
    const center = typeCenters.get(typeOrCat)?.clone() ?? new THREE.Vector3(0, 0, 0);
    center.x += (Math.random() - 0.5) * 4;
    center.y += (Math.random() - 0.5) * 3;
    center.z += (Math.random() - 0.5) * 4;
    nodePositions.set(g.name, center);
    nodeVelocities.set(g.name, new THREE.Vector3());
  });
}

function stepLayout(dt) {
  if (!glyphData.length) return;

  const forces = new Map();
  glyphData.forEach(g => forces.set(g.name, new THREE.Vector3()));

  // Semantic springs
  edgesSemantic.forEach(edge => {
    const pa = nodePositions.get(edge.source), pb = nodePositions.get(edge.target);
    if (!pa || !pb) return;
    const dir = new THREE.Vector3().subVectors(pb, pa);
    const dist = dir.length() + 1e-4;
    dir.normalize();
    const k = LAYOUT_PARAMS.springSemantic * (edge.weight || 1.0) * (1 - CORTEX_LAYOUT.layoutBlend);
    const f = dir.clone().multiplyScalar(k * dist);
    forces.get(edge.source)?.add(f);
    forces.get(edge.target)?.sub(f);
  });

  // Neural springs
  edgesNeural.forEach(edge => {
    const pa = nodePositions.get(edge.source), pb = nodePositions.get(edge.target);
    if (!pa || !pb) return;
    const dir = new THREE.Vector3().subVectors(pb, pa);
    const dist = dir.length() + 1e-4;
    dir.normalize();
    const boosted = Math.max(edge.weight || 0, 0.05);
    const k = LAYOUT_PARAMS.springNeural * boosted * (1 - CORTEX_LAYOUT.layoutBlend);
    const f = dir.clone().multiplyScalar(k * dist);
    forces.get(edge.source)?.add(f);
    forces.get(edge.target)?.sub(f);
  });

  // Hierarchy springs (taxonomy edges get 15x strength)
  edgesHierarchy.forEach(edge => {
    const pa = nodePositions.get(edge.source), pb = nodePositions.get(edge.target);
    if (!pa || !pb) return;
    const dir = new THREE.Vector3().subVectors(pb, pa);
    const dist = dir.length() + 1e-4;
    dir.normalize();
    const typeA = nodeTypeByName.get(edge.source) || 'concept';
    const typeB = nodeTypeByName.get(edge.target) || 'concept';
    const isTaxEdge = typeA === 'taxonomy' || typeB === 'taxonomy';
    const kBase = (isTaxEdge ? LAYOUT_PARAMS.springHierarchy * 15.0 : LAYOUT_PARAMS.springHierarchy) * (edge.weight || 1.0);
    const k = kBase * (1 - CORTEX_LAYOUT.layoutBlend);
    const f = dir.clone().multiplyScalar(k * dist);
    forces.get(edge.source)?.add(f);
    forces.get(edge.target)?.sub(f);
  });

  // Cortex rest-length springs (dominant when layoutBlend=1)
  if (CORTEX_LAYOUT.enabled && edgesCortex.length && CORTEX_LAYOUT.layoutBlend > 0) {
    edgesCortex.forEach(edge => {
      const pa = nodePositions.get(edge.source), pb = nodePositions.get(edge.target);
      if (!pa || !pb) return;
      const dir = new THREE.Vector3().subVectors(pb, pa);
      const dist = dir.length() + 1e-4;
      dir.normalize();
      const k = CORTEX_LAYOUT.springK * edge.weight * CORTEX_LAYOUT.layoutBlend;
      const stretch = dist - edge.rest;
      const f = dir.clone().multiplyScalar(k * stretch);
      forces.get(edge.source)?.add(f);
      forces.get(edge.target)?.sub(f);
    });
  }

  // Repulsion (all pairs)
  const names = glyphData.map(g => g.name);
  const repulsion = LAYOUT_PARAMS.repulsion * (CORTEX_LAYOUT.enabled ? CORTEX_LAYOUT.repulsionBoost : 1.0);

  for (let i = 0; i < names.length; i++) {
    for (let j = i + 1; j < names.length; j++) {
      const pi = nodePositions.get(names[i]), pj = nodePositions.get(names[j]);
      if (!pi || !pj) continue;
      const dir = new THREE.Vector3().subVectors(pj, pi);
      const dist2 = dir.lengthSq() + 1e-4;
      dir.normalize();
      const forceMag = repulsion / dist2;
      const f = dir.clone().multiplyScalar(forceMag);
      forces.get(names[i])?.sub(f);
      forces.get(names[j])?.add(f);
    }
  }

  // Taxonomy center gravity
  glyphData.forEach(g => {
    if (nodeTypeByName.get(g.name) !== 'taxonomy') return;
    const pos = nodePositions.get(g.name);
    if (!pos) return;
    const dir = pos.clone().multiplyScalar(-1);
    const dist = dir.length() + 1e-4;
    dir.normalize();
    const force = Math.min(12.0, 2.8 * dist);
    forces.get(g.name)?.add(dir.multiplyScalar(force));
  });

  // Integrate
  glyphData.forEach(g => {
    const pos = nodePositions.get(g.name);
    const vel = nodeVelocities.get(g.name);
    const f = forces.get(g.name);
    if (!pos || !vel || !f) return;
    vel.addScaledVector(f, dt);
    vel.multiplyScalar(LAYOUT_PARAMS.damping);
    const step = vel.clone().multiplyScalar(dt);
    if (step.length() > LAYOUT_PARAMS.maxStep) step.setLength(LAYOUT_PARAMS.maxStep);
    pos.add(step);
  });

  // Recenter centroid to origin
  const center = new THREE.Vector3();
  glyphData.forEach(g => { const p = nodePositions.get(g.name); if (p) center.add(p); });
  center.multiplyScalar(1 / glyphData.length);
  glyphData.forEach(g => { const p = nodePositions.get(g.name); if (p) p.sub(center); });
}

// ── Scene Building ──

function rebuildScene() {
  if (!scene) return;

  if (labelsEl) labelsEl.innerHTML = '';
  glyphGroups.forEach(group => scene.remove(group));
  glyphGroups = [];
  glyphByName.clear();

  // Remove old edge lines
  [semanticLines, neuralLines, hierarchyLines,
   brightLinesSemantic, brightLinesNeural, brightLinesHierarchy].forEach(l => {
    if (l) scene.remove(l);
  });
  semanticLines = neuralLines = hierarchyLines = null;
  brightLinesSemantic = brightLinesNeural = brightLinesHierarchy = null;

  if (!glyphData.length) return;

  glyphData.forEach(g => {
    const group = createGlyphGroup(g);
    group.scale.set(0.2, 0.2, 0.2);
    const pos = nodePositions.get(g.name) || new THREE.Vector3();
    group.position.copy(pos);
    scene.add(group);
    glyphGroups.push(group);
    glyphByName.set(g.name, group);
  });

  buildEdgesOnce();
  updateHighlightStyles();
}

function createGlyphGroup(glyph) {
  const group = new THREE.Group();
  group.userData.name = glyph.name;
  group.userData.node_type = glyph.node_type || 'concept';

  let layer = glyph.layers?.find(l => l.index === activeLayer);
  if (!layer && glyph.layers) layer = glyph.layers.find(l => l.index === 0);
  if (!layer) return group;

  const cortexBits = layer.cortex || [];
  const nodeType = glyph.node_type || 'concept';

  if (nodeType === 'taxonomy') {
    // Taxonomy: single green point
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute([0, 0, 0], 3));
    const material = new THREE.PointsMaterial({ color: 0x39ff14, size: 0.06, transparent: true });
    group.add(new THREE.Points(geometry, material));
  } else {
    // Cortex point cloud from boolean bits
    const cortex = buildPointCloud(cortexBits, 0.41, new THREE.Color(0x00ffea));
    group.add(cortex);
  }

  // HTML label
  if (labelsEl) {
    let label = labelsEl.querySelector(`[data-glyph-label="${CSS.escape(glyph.name)}"]`);
    if (!label) {
      label = document.createElement('div');
      label.dataset.glyphLabel = glyph.name;
      label.innerText = glyph.name;
      label.style.cssText = 'position:absolute;color:#fff;font-size:12px;font-weight:700;' +
        'text-shadow:0 0 8px rgba(0,0,0,0.95);background:rgba(0,0,0,0.55);' +
        'padding:2px 4px;border-radius:6px;pointer-events:none;transform:translate(-50%,-100%);' +
        'white-space:nowrap;';
      labelsEl.appendChild(label);
    }
    group.userData.label = label;
  }

  return group;
}

// ── Point Cloud ──

function buildPointCloud(bits, scale, color) {
  const geometry = new THREE.BufferGeometry();
  const count = bits.length || 0;
  const dim = count > 0 ? Math.ceil(Math.cbrt(count)) : 1;

  const positions = [];
  const colors = [];

  for (let i = 0; i < count; i++) {
    if (!bits[i]) continue;
    const ix = i % dim;
    const iy = Math.floor(i / dim) % dim;
    const iz = Math.floor(i / (dim * dim));
    const x = (ix - dim / 2) * (3 / dim) * scale;
    const y = (iy - dim / 2) * (3 / dim) * scale;
    const z = (iz - dim / 2) * (3 / dim) * scale;
    positions.push(x, y, z);
    colors.push(color.r, color.g, color.b);
  }

  if (positions.length === 0) {
    positions.push(0, 0, 0);
    const dimColor = color.clone().multiplyScalar(0.3);
    colors.push(dimColor.r, dimColor.g, dimColor.b);
  }

  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({ size: 0.02 * scale, vertexColors: true, transparent: true });
  return new THREE.Points(geometry, material);
}

// ── Edges ──

function buildEdgesOnce() {
  if (!scene) return;

  if (edgesSemantic.length) {
    const positions = new Float32Array(edgesSemantic.length * 2 * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({ color: 0xffaa00, opacity: 0.6, transparent: true, depthTest: false });
    semanticLines = new THREE.LineSegments(geometry, material);
    scene.add(semanticLines);
  }

  if (edgesNeural.length) {
    const positions = new Float32Array(edgesNeural.length * 2 * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({ color: 0x00aaff, opacity: 0.5, transparent: true, depthTest: false });
    neuralLines = new THREE.LineSegments(geometry, material);
    scene.add(neuralLines);
  }

  if (edgesHierarchy.length) {
    const positions = new Float32Array(edgesHierarchy.length * 2 * 3);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const material = new THREE.LineBasicMaterial({ color: 0xdd66ff, opacity: 0.7, transparent: true, depthTest: false });
    hierarchyLines = new THREE.LineSegments(geometry, material);
    scene.add(hierarchyLines);
  }

  updateEdgePositions();
}

function updateEdgePositions() {
  if (!scene) return;

  const updateLine = (line, edges) => {
    if (!line || !edges.length) return;
    const posAttr = line.geometry.getAttribute('position');
    const arr = posAttr.array;
    let idx = 0;

    edges.forEach(edge => {
      const srcGroup = glyphByName.get(edge.source);
      const dstGroup = glyphByName.get(edge.target);
      if (!srcGroup || !dstGroup) return;
      const srcPos = new THREE.Vector3();
      const dstPos = new THREE.Vector3();
      srcGroup.getWorldPosition(srcPos);
      dstGroup.getWorldPosition(dstPos);
      arr[idx++] = srcPos.x; arr[idx++] = srcPos.y; arr[idx++] = srcPos.z;
      arr[idx++] = dstPos.x; arr[idx++] = dstPos.y; arr[idx++] = dstPos.z;
    });

    posAttr.needsUpdate = true;
  };

  updateLine(semanticLines, edgesSemantic);
  updateLine(neuralLines, edgesNeural);
  updateLine(hierarchyLines, edgesHierarchy);
}

function updateBrightEdgePositions() {
  if (!highlightMode) return;

  const updateBright = (line, edges) => {
    if (!line) return;
    const posAttr = line.geometry.getAttribute('position');
    const arr = posAttr.array;
    let idx = 0;
    edges.forEach(e => {
      if (!edgeIsHighlighted(e)) return;
      const src = glyphByName.get(e.source);
      const dst = glyphByName.get(e.target);
      if (!src || !dst) return;
      const s = new THREE.Vector3(), d = new THREE.Vector3();
      src.getWorldPosition(s);
      dst.getWorldPosition(d);
      arr[idx++] = s.x; arr[idx++] = s.y; arr[idx++] = s.z;
      arr[idx++] = d.x; arr[idx++] = d.y; arr[idx++] = d.z;
    });
    posAttr.needsUpdate = true;
  };

  if (brightLinesSemantic && showSemantic) updateBright(brightLinesSemantic, edgesSemantic);
  if (brightLinesNeural && showNeural) updateBright(brightLinesNeural, edgesNeural);
  if (brightLinesHierarchy && showHierarchy) updateBright(brightLinesHierarchy, edgesHierarchy);
}

// ── Labels ──

function updateLabels() {
  if (!camera || !showLabels) {
    glyphGroups.forEach(group => {
      const label = group.userData.label;
      if (label) label.style.display = 'none';
    });
    return;
  }

  const container = containerEl;
  const w = container ? container.clientWidth : window.innerWidth;
  const h = container ? container.clientHeight : window.innerHeight;

  glyphGroups.forEach(group => {
    const label = group.userData.label;
    if (!label) return;

    group.getWorldPosition(_worldPos);
    _worldPos.y += 0.18;

    if (highlightMode && !highlightSet.has(group.userData.name)) {
      label.style.display = 'none';
      return;
    }

    const projected = _worldPos.clone().project(camera);
    if (projected.z > 1) {
      label.style.display = 'none';
      return;
    }

    label.style.display = 'block';
    const x = (projected.x * 0.5 + 0.5) * w;
    const y = (-projected.y * 0.5 + 0.5) * h;
    label.style.left = `${x}px`;
    label.style.top = `${y}px`;
    label.style.transform = 'translate(-50%, -70%)';
    label.style.fontSize = '11px';

    const name = group.userData.name;
    if (highlightMode) {
      if (highlightSet.has(name)) {
        label.style.color = '#00ffd0';
        label.style.fontWeight = '600';
        label.style.opacity = '1.0';
      } else {
        label.style.color = '#cccccc';
        label.style.fontWeight = '400';
        label.style.opacity = '0.25';
      }
    } else if (name === selectedName) {
      label.style.color = '#00ffd0';
      label.style.fontWeight = '600';
      label.style.opacity = '1.0';
    } else {
      label.style.color = '#ffffff';
      label.style.fontWeight = '400';
      label.style.opacity = '1.0';
    }
  });
}

// ── Highlight System ──

function edgeIsHighlighted(edge) {
  return highlightSet.has(edge.source) && highlightSet.has(edge.target);
}

function computeHighlightSet(baseName) {
  if (!baseName) return new Set();
  const next = new Set([baseName]);

  const semEnabled = highlightMode ? highlightShowSemantic : showSemantic;
  const neuEnabled = highlightMode ? highlightShowNeural : showNeural;
  const hierEnabled = highlightMode ? highlightShowHierarchy : showHierarchy;

  if (semEnabled) {
    edgesSemantic.forEach(e => {
      if (e.source === baseName) next.add(e.target);
      if (e.target === baseName) next.add(e.source);
    });
  }
  if (neuEnabled) {
    edgesNeural.forEach(e => {
      if (e.source === baseName) next.add(e.target);
      if (e.target === baseName) next.add(e.source);
    });
  }
  if (hierEnabled) {
    edgesHierarchy.forEach(e => {
      if (e.source === baseName) next.add(e.target);
      if (e.target === baseName) next.add(e.source);
    });
  }

  if (!semEnabled && !neuEnabled && !hierEnabled) return new Set([baseName]);
  return next;
}

function updateHighlightStyles() {
  if (!scene) return;
  const dimColor = new THREE.Color(0x444444);

  glyphGroups.forEach(group => {
    const name = group.userData.name;
    const isBright = highlightMode && highlightSet.has(name);

    group.visible = !highlightMode || isBright;
    const label = group.userData.label;
    if (label) label.style.display = group.visible ? 'block' : 'none';
    if (!group.visible) return;

    group.traverse(obj => {
      const mat = obj.material;
      if (!mat || !mat.color) return;
      if (!mat._originalColor) mat._originalColor = mat.color.clone();

      if (!highlightMode) {
        mat.color.copy(mat._originalColor);
        mat.opacity = 1.0;
        mat.transparent = true;
        return;
      }

      if (isBright) {
        mat.color.copy(mat._originalColor);
        mat.opacity = 1.0;
        mat.transparent = true;
      } else {
        mat.color.copy(dimColor);
        mat.opacity = 0.15;
        mat.transparent = true;
      }
    });
  });

  // Non-highlight mode: show/hide base edges
  if (!highlightMode) {
    if (semanticLines) {
      semanticLines.visible = showSemantic;
      if (showSemantic) { semanticLines.material.color.set(0xffaa00); semanticLines.material.opacity = 0.6; }
    }
    if (neuralLines) {
      neuralLines.visible = showNeural;
      if (showNeural) { neuralLines.material.color.set(0x00aaff); neuralLines.material.opacity = 0.5; }
    }
    if (hierarchyLines) {
      hierarchyLines.visible = showHierarchy;
      if (showHierarchy) { hierarchyLines.material.color.set(0xdd66ff); hierarchyLines.material.opacity = 0.7; }
    }

    // Remove bright edge overlays
    [brightLinesSemantic, brightLinesNeural, brightLinesHierarchy].forEach(l => {
      if (l) scene.remove(l);
    });
    brightLinesSemantic = brightLinesNeural = brightLinesHierarchy = null;
    return;
  }

  // Highlight mode: hide base edges, build bright overlays
  if (semanticLines) semanticLines.visible = false;
  if (neuralLines) neuralLines.visible = false;
  if (hierarchyLines) hierarchyLines.visible = false;

  const buildBrightEdgeLayer = (baseLines, edges, color, kind) => {
    let prev = kind === 'semantic' ? brightLinesSemantic
      : kind === 'neural' ? brightLinesNeural : brightLinesHierarchy;
    if (prev) scene.remove(prev);

    if (!highlightMode || !baseLines) {
      if (kind === 'semantic') brightLinesSemantic = null;
      if (kind === 'neural') brightLinesNeural = null;
      if (kind === 'hierarchy') brightLinesHierarchy = null;
      return;
    }

    baseLines.material.color.set(0x444444);
    baseLines.material.opacity = 0.08;

    const brightEdges = edges.filter(e => edgeIsHighlighted(e));
    if (!brightEdges.length) {
      if (kind === 'semantic') brightLinesSemantic = null;
      if (kind === 'neural') brightLinesNeural = null;
      if (kind === 'hierarchy') brightLinesHierarchy = null;
      return;
    }

    const positions = [];
    brightEdges.forEach(e => {
      const src = glyphByName.get(e.source);
      const dst = glyphByName.get(e.target);
      if (!src || !dst) return;
      const s = new THREE.Vector3(), d = new THREE.Vector3();
      src.getWorldPosition(s);
      dst.getWorldPosition(d);
      positions.push(s.x, s.y, s.z, d.x, d.y, d.z);
    });

    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const mat = new THREE.LineBasicMaterial({ color, opacity: 1.0, transparent: true, depthTest: false });
    const ls = new THREE.LineSegments(geom, mat);
    scene.add(ls);

    if (kind === 'semantic') brightLinesSemantic = ls;
    if (kind === 'neural') brightLinesNeural = ls;
    if (kind === 'hierarchy') brightLinesHierarchy = ls;
  };

  if (!highlightShowSemantic) {
    if (brightLinesSemantic) { scene.remove(brightLinesSemantic); brightLinesSemantic = null; }
  } else buildBrightEdgeLayer(semanticLines, edgesSemantic, 0xffaa00, 'semantic');

  if (!highlightShowNeural) {
    if (brightLinesNeural) { scene.remove(brightLinesNeural); brightLinesNeural = null; }
  } else buildBrightEdgeLayer(neuralLines, edgesNeural, 0x00aaff, 'neural');

  if (!highlightShowHierarchy) {
    if (brightLinesHierarchy) { scene.remove(brightLinesHierarchy); brightLinesHierarchy = null; }
  } else buildBrightEdgeLayer(hierarchyLines, edgesHierarchy, 0xdd66ff, 'hierarchy');
}

// ── Search & Highlight ──

function clearHighlight() {
  highlightMode = false;
  highlightSet.clear();
  selectedName = null;
  highlightShowSemantic = showSemantic;
  highlightShowNeural = showNeural;
  highlightShowHierarchy = showHierarchy;
  updateHighlightStyles();
}

function selectByName(queryRaw) {
  const query = queryRaw.toLowerCase();
  if (!glyphData.length) return;

  let found = null;
  glyphData.forEach(g => { if (g.name.toLowerCase() === query) found = g.name; });
  if (!found) {
    glyphData.forEach(g => { if (!found && g.name.toLowerCase().startsWith(query)) found = g.name; });
  }
  if (!found) { clearHighlight(); return; }

  selectedName = found;
  highlightMode = true;
  highlightShowSemantic = showSemantic;
  highlightShowNeural = showNeural;
  highlightShowHierarchy = showHierarchy;
  highlightSet = computeHighlightSet(found);

  // Focus camera on selected node
  const group = glyphByName.get(found);
  if (group && controls) {
    const pos = new THREE.Vector3();
    group.getWorldPosition(pos);
    controls.target.copy(pos);
  }

  updateHighlightStyles();
}

function applyChatHighlight(detail) {
  const names = new Set((detail.highlightNames || []).filter(Boolean));
  const glyphName = detail.glyph || (names.size ? [...names][0] : null);
  if (glyphName) names.add(glyphName);
  if (names.size === 0) { clearHighlight(); return; }

  selectedName = glyphName;
  highlightMode = true;
  highlightShowSemantic = showSemantic;
  highlightShowNeural = showNeural;
  highlightShowHierarchy = showHierarchy;
  highlightSet = new Set(names);

  if (glyphName) {
    const group = glyphByName.get(glyphName);
    if (group && controls) {
      const pos = new THREE.Vector3();
      group.getWorldPosition(pos);
      controls.target.copy(pos);
    }
  }

  updateHighlightStyles();
}

// ── Event Handlers ──

function onKeyDown(e) {
  // Ignore when focus is in an input
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

  // Layer switching
  if (e.key === '1') activeLayer = 0;
  if (e.key === '2') activeLayer = 1;
  if (e.key === '3') activeLayer = 2;
  if (e.key === '4') activeLayer = 3;

  if (e.key === 'e' || e.key === 'E') explodeMode = !explodeMode;
  if (e.key === 'Escape') clearHighlight();
  if (e.key === 'f' || e.key === 'F') {
    frozen = !frozen;
    refreshOverlayText();
  }
  if (e.key === 'l' || e.key === 'L') {
    showLabels = !showLabels;
    if (labelsEl) labelsEl.style.display = showLabels ? 'block' : 'none';
    updateCheckboxState();
  }

  // On layer change: rebuild cortex bits + edges
  if (e.key === '1' || e.key === '2' || e.key === '3' || e.key === '4') {
    cortexBitsByName.clear();
    glyphData.forEach(g => cortexBitsByName.set(g.name, getLayerCortexBits(g, activeLayer)));
    initLayout();
    if (CORTEX_LAYOUT.enabled) buildCortexEdgesKNN(CORTEX_LAYOUT.K);
  }

  rebuildScene();
  updateLayerKeyUI();
}

function onToggle(ev) {
  const d = ev.detail || {};
  if (typeof d.semantic === 'boolean') showSemantic = d.semantic;
  if (typeof d.neural === 'boolean') showNeural = d.neural;
  if (typeof d.hierarchy === 'boolean') showHierarchy = d.hierarchy;
  if (typeof d.labels === 'boolean') {
    showLabels = d.labels;
    if (labelsEl) labelsEl.style.display = showLabels ? 'block' : 'none';
  }
  if (highlightMode && selectedName) highlightSet = computeHighlightSet(selectedName);
  updateHighlightStyles();
  updateLayerKeyUI();
}

function onGlyphSelect(ev) {
  const glyph = ev.detail;
  if (!glyph) { clearHighlight(); return; }
  const name = glyph.concept_text || glyph.name || glyph.id;
  if (name) selectByName(name);
}

function onResultHighlight(ev) {
  const detail = ev.detail;
  if (!detail || !detail.ft) return;

  const matchNames = new Set();
  const ft = detail.ft;
  if (ft?.children) {
    for (const child of ft.children) {
      if (child.description === 'results' && Array.isArray(child.children)) {
        for (const match of child.children) {
          const name = match.value?.concept_text;
          if (name) matchNames.add(name);
        }
      }
    }
  }

  if (matchNames.size > 0) {
    applyChatHighlight({ highlightNames: [...matchNames] });
  }
}

// ── Animation Loop ──

function animate() {
  animationId = requestAnimationFrame(animate);

  if (!frozen) {
    stepLayout(0.02);
    updateNodePositions();
    updateEdgePositions();
    updateBrightEdgePositions();
  }
  updateLabels();

  controls?.update();
  renderer?.render(scene, camera);
}

function updateNodePositions() {
  glyphGroups.forEach(group => {
    const name = group.userData?.name;
    if (!name) return;
    const basePos = nodePositions.get(name);
    if (!basePos) return;
    if (!explodeMode) group.position.copy(basePos);
    else group.position.copy(basePos).multiplyScalar(EXPLODE_SCALE);
  });
}

// ── UI Overlays ──

function buildUI() {
  removeUI();

  // Overlay text (top-left)
  overlayEl = document.createElement('div');
  overlayEl.id = 'viewer-overlay-text';
  overlayEl.className = 'viewer-ui-panel viewer-ui-topleft';
  overlayEl.textContent = OVERLAY_TEXT;
  containerEl.appendChild(overlayEl);

  // Layer key (top-left, below overlay)
  layerKeyEl = document.createElement('div');
  layerKeyEl.id = 'viewer-layer-key';
  layerKeyEl.className = 'viewer-ui-panel viewer-ui-layerkey';
  containerEl.appendChild(layerKeyEl);

  // Edge toggles (top-right)
  edgeTogglesEl = document.createElement('div');
  edgeTogglesEl.id = 'viewer-edge-toggles';
  edgeTogglesEl.className = 'viewer-ui-panel viewer-ui-topright';
  edgeTogglesEl.innerHTML = `
    <b>EDGES</b><br>
    <label><input type="checkbox" id="chk-semantic" checked> Semantic</label><br>
    <label><input type="checkbox" id="chk-neural" checked> Neural</label><br>
    <label><input type="checkbox" id="chk-hierarchy" checked> Hierarchy</label><br>
    <div style="margin-top:6px;border-top:1px solid rgba(255,255,255,0.2);padding-top:6px;">
      <b>VIEW</b><br>
      <label><input type="checkbox" id="chk-labels" checked> Labels</label>
    </div>`;
  containerEl.appendChild(edgeTogglesEl);

  // Wire checkbox events
  const wire = (id, prop) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('change', () => {
      const checked = el.checked;
      if (prop === 'labels') {
        showLabels = checked;
        if (labelsEl) labelsEl.style.display = checked ? 'block' : 'none';
      } else {
        if (prop === 'semantic') showSemantic = checked;
        if (prop === 'neural') showNeural = checked;
        if (prop === 'hierarchy') showHierarchy = checked;
      }
      if (highlightMode && selectedName) highlightSet = computeHighlightSet(selectedName);
      updateHighlightStyles();
      updateLayerKeyUI();
    });
  };
  wire('chk-semantic', 'semantic');
  wire('chk-neural', 'neural');
  wire('chk-hierarchy', 'hierarchy');
  wire('chk-labels', 'labels');

  // Search panel (bottom-right)
  searchPanelEl = document.createElement('div');
  searchPanelEl.id = 'viewer-search-panel';
  searchPanelEl.className = 'viewer-ui-panel viewer-ui-bottomright';
  searchPanelEl.innerHTML = `
    <b>SEARCH</b><br>
    <input type="text" id="viewer-search-input" placeholder="Concept name\u2026"
           style="width:180px;padding:3px 6px;border:1px solid rgba(255,255,255,0.5);border-radius:4px;background:#000;color:#fff;font-size:11px;">
    <div style="font-size:9px;margin-top:3px;color:#aaa;">(Exact or prefix match, press Enter)</div>`;
  containerEl.appendChild(searchPanelEl);

  const searchInput = document.getElementById('viewer-search-input');
  if (searchInput) {
    searchInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') {
        const q = searchInput.value.trim();
        if (!q) clearHighlight();
        else selectByName(q);
      }
      e.stopPropagation(); // prevent keydown from reaching viewer shortcuts
    });
  }

  // Legend (bottom-left)
  legendEl = document.createElement('div');
  legendEl.id = 'viewer-legend-panel';
  legendEl.className = 'viewer-ui-panel viewer-ui-bottomleft';
  containerEl.appendChild(legendEl);
  buildLegend();
}

function removeUI() {
  ['viewer-overlay-text', 'viewer-layer-key', 'viewer-edge-toggles',
   'viewer-search-panel', 'viewer-legend-panel'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.remove();
  });
  overlayEl = layerKeyEl = legendEl = edgeTogglesEl = searchPanelEl = null;
}

function refreshOverlayText() {
  if (overlayEl) overlayEl.textContent = `${OVERLAY_TEXT}${frozen ? ' (frozen)' : ''}`;
}

function updateLayerKeyUI() {
  if (!layerKeyEl) return;
  let labelNames = ['Physical', 'Semantic', 'Functional', 'Linguistic'];
  if (rolesConfig?.layers) labelNames = rolesConfig.layers.map(l => l.name || `Layer ${l.index}`);

  layerKeyEl.innerHTML = `
    <b>LAYER SELECT</b><br>
    ${activeLayer === 0 ? '&gt; ' : ''}1 \u2013 ${labelNames[0] || 'Layer 0'}<br>
    ${activeLayer === 1 ? '&gt; ' : ''}2 \u2013 ${labelNames[1] || 'Layer 1'}<br>
    ${activeLayer === 2 ? '&gt; ' : ''}3 \u2013 ${labelNames[2] || 'Layer 2'}<br>
    ${activeLayer === 3 ? '&gt; ' : ''}4 \u2013 ${labelNames[3] || 'Layer 3'}<br>
    <br>
    [E] Explode: ${explodeMode ? 'ON' : 'OFF'}<br>
    [F] Frozen: ${frozen ? 'ON' : 'OFF'}<br>
    [L] Labels: ${showLabels ? 'ON' : 'OFF'}<br>
    [Esc] Reset highlight<br>
    <br>
    Cortex layout: ${CORTEX_LAYOUT.enabled ? 'ON' : 'OFF'} (k=${CORTEX_LAYOUT.K})`;
}

function updateCheckboxState() {
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.checked = val; };
  set('chk-semantic', showSemantic);
  set('chk-neural', showNeural);
  set('chk-hierarchy', showHierarchy);
  set('chk-labels', showLabels);
}

function buildLegend() {
  if (!legendEl) return;
  const seen = new Set();
  glyphData.forEach(g => { const t = g.node_type; if (t) seen.add(t); });
  if (seen.size === 0) { legendEl.style.display = 'none'; return; }

  let html = '';
  for (const t of [...seen].sort()) {
    const c = TYPE_COLORS_CSS[t.toLowerCase()] || '#64748b';
    html += `<div style="display:flex;align-items:center;gap:5px;">` +
      `<span style="width:8px;height:8px;border-radius:50%;background:${c};flex-shrink:0;"></span>${t}</div>`;
  }
  legendEl.innerHTML = html;
}

// ── Cleanup ──

function cleanup() {
  if (animationId) { cancelAnimationFrame(animationId); animationId = null; }
  window.removeEventListener('resize', onWindowResize);
  window.removeEventListener('keydown', onKeyDown);

  if (controls) controls.dispose();

  if (labelsEl) labelsEl.innerHTML = '';

  const disposeObject = obj => {
    if (obj.geometry) obj.geometry.dispose?.();
    if (obj.material) {
      if (Array.isArray(obj.material)) obj.material.forEach(m => m.dispose?.());
      else obj.material.dispose?.();
    }
  };

  if (scene) {
    scene.traverse(child => disposeObject(child));
    while (scene.children.length > 0) scene.remove(scene.children[0]);
  }

  [semanticLines, neuralLines, hierarchyLines,
   brightLinesSemantic, brightLinesNeural, brightLinesHierarchy].forEach(l => {
    if (l) { disposeObject(l); if (scene) scene.remove(l); }
  });

  semanticLines = neuralLines = hierarchyLines = null;
  brightLinesSemantic = brightLinesNeural = brightLinesHierarchy = null;

  glyphData = [];
  edgesSemantic = [];
  edgesNeural = [];
  edgesHierarchy = [];
  edgesCortex = [];
  cortexBitsByName.clear();
  nodePositions.clear();
  nodeVelocities.clear();
  glyphGroups = [];
  glyphByName.clear();
  nodeTypeByName.clear();

  highlightMode = false;
  highlightSet.clear();
  selectedName = null;

  if (renderer) {
    renderer.dispose();
    renderer.forceContextLoss?.();
  }
  if (mountEl) mountEl.innerHTML = '';

  scene = camera = renderer = controls = null;
  removeUI();
}
