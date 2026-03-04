/**
 * Auth module — Platform JWT login, token management.
 *
 * Non-local deployments authenticate with the Platform.
 * Local mode skips auth entirely.
 */

const STORAGE_KEY_TOKEN = 'glyphh_token';
const STORAGE_KEY_USER = 'glyphh_user';

/** Get stored JWT token. */
export function getToken() {
  return localStorage.getItem(STORAGE_KEY_TOKEN);
}

/** Get stored user info. */
export function getUser() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY_USER);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

/** Check if user is authenticated (has a token). */
export function isAuthenticated() {
  const cfg = window.__GLYPHH__ || {};
  if (cfg.isLocal) return true;
  return !!getToken();
}

/** Build Authorization headers for API calls. */
export function authHeaders() {
  const cfg = window.__GLYPHH__ || {};
  if (cfg.isLocal) return {};
  const token = getToken();
  if (!token) return {};
  return { 'Authorization': `Bearer ${token}` };
}

/**
 * Login via Platform.
 * POST to Platform /auth/login → { access_token, user }
 * Store token + user in localStorage.
 */
export async function login(email, password) {
  const cfg = window.__GLYPHH__ || {};
  const platformUrl = cfg.platformUrl || 'https://api.glyphh.ai/api/v1';

  const res = await fetch(`${platformUrl}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
  });

  if (!res.ok) {
    let detail = 'Login failed';
    try {
      const data = await res.json();
      detail = data.detail || data.message || detail;
    } catch {}
    throw new Error(detail);
  }

  const data = await res.json();
  const token = data.access_token;
  const user = data.user || {};

  if (!token) throw new Error('No token in response');

  localStorage.setItem(STORAGE_KEY_TOKEN, token);
  localStorage.setItem(STORAGE_KEY_USER, JSON.stringify(user));

  return { token, user };
}

/** Clear auth state and redirect to login. */
export function logout() {
  localStorage.removeItem(STORAGE_KEY_TOKEN);
  localStorage.removeItem(STORAGE_KEY_USER);
  window.location.href = '/login';
}
