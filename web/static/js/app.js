/**
 * Main app controller — initializes all panels, handles routing.
 */

import { isAuthenticated, logout } from './auth.js';
import { init as initChat } from './chat.js';
import { init as initGlyphs } from './glyphs.js';
import { init as initViewer } from './viewer.js';
import { init as initDropzone } from './dropzone.js';
import { init as initResults } from './results.js';
import { init as initResize } from './resize.js';

const cfg = window.__GLYPHH__;

// Auth check (non-local only)
if (!cfg.isLocal && !isAuthenticated()) {
  window.location.href = '/login';
} else {
  // Initialize all panels
  initChat();
  initGlyphs();
  initViewer();
  initDropzone();
  initResults();
  initResize();

  // Logout button
  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', logout);
  }
}
