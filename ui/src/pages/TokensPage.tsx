import { useEffect, useState, useCallback } from "react";
import { useAuth } from "../context/AuthContext";
import PageHeader from "../components/layout/PageHeader";
import ConfirmDialog from "../components/ConfirmDialog";
import { HiOutlineKey, HiOutlineTrash, HiOutlineDuplicate, HiCheck } from "react-icons/hi";

type Token = {
  id: string;
  name: string;
  token_prefix: string;
  model_id?: string | null;
  permissions: string[];
  created_at?: string;
  expires_at?: string | null;
  status: string;
};

type CreateResponse = {
  id: string;
  name: string;
  token: string;
  token_prefix: string;
  org_id: string;
  model_id?: string | null;
  permissions: string[];
  expires_at?: string | null;
};

export default function TokensPage() {
  const { api, orgId } = useAuth();
  const [tokens, setTokens] = useState<Token[]>([]);
  const [loading, setLoading] = useState(true);
  const [showCreate, setShowCreate] = useState(false);
  const [createdToken, setCreatedToken] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Create form state
  const [formName, setFormName] = useState("");
  const [formModelId, setFormModelId] = useState("");
  const [formPerms, setFormPerms] = useState<Set<string>>(new Set(["read", "write"]));
  const [formExpiry, setFormExpiry] = useState("365");
  const [creating, setCreating] = useState(false);
  const [revokeTarget, setRevokeTarget] = useState<{ id: string; name: string } | null>(null);

  const loadTokens = async () => {
    if (!orgId) return;
    setLoading(true);
    try {
      const data = await api.get<{ tokens: Token[] }>(`/${orgId}/tokens`);
      setTokens(data.tokens || []);
    } catch {
      setTokens([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadTokens();
  }, [orgId]);

  const handleCreate = async () => {
    if (!orgId || !formName.trim() || creating) return;
    setCreating(true);
    try {
      const data = await api.post<CreateResponse>(`/${orgId}/tokens`, {
        name: formName.trim(),
        model_id: formModelId.trim() || null,
        permissions: Array.from(formPerms),
        expires_days: formExpiry === "never" ? null : parseInt(formExpiry),
      });
      setCreatedToken(data.token);
      setCopied(false);
      await loadTokens();
    } catch (e: unknown) {
      alert("Create failed: " + (e instanceof Error ? e.message : ""));
    } finally {
      setCreating(false);
    }
  };

  const executeRevoke = useCallback(async () => {
    if (!orgId || !revokeTarget) return;
    setRevokeTarget(null);
    try {
      await api.delete(`/${orgId}/tokens/${revokeTarget.id}`);
      await loadTokens();
    } catch (e: unknown) {
      alert("Revoke failed: " + (e instanceof Error ? e.message : ""));
    }
  }, [orgId, revokeTarget, api]);

  const togglePerm = (perm: string) => {
    setFormPerms((prev) => {
      const next = new Set(prev);
      if (next.has(perm)) next.delete(perm);
      else next.add(perm);
      return next;
    });
  };

  const resetForm = () => {
    setFormName("");
    setFormModelId("");
    setFormPerms(new Set(["read", "write"]));
    setFormExpiry("365");
    setCreatedToken(null);
    setCopied(false);
    setShowCreate(false);
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title={
          <span className="inline-flex items-center gap-2">
            <HiOutlineKey size={22} className="text-[var(--accent)]" />
            <span>API Tokens</span>
          </span>
        }
        subtitle="Manage tokens for listeners, MCP clients, and external integrations."
        actions={
          <button
            onClick={() => { setCreatedToken(null); setShowCreate(true); }}
            className="px-3 py-2 text-sm font-semibold rounded-xl bg-[var(--accent)] btn-primary shadow shadow-[var(--accent)]/30 hover:opacity-95 transition cursor-pointer"
          >
            Create Token
          </button>
        }
      />

      {/* Token table */}
      <div className="px-4">
        <div className="border border-[var(--border)] rounded-xl overflow-hidden bg-[var(--card-bg)]">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[var(--border)]">
                <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Name</th>
                <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Prefix</th>
                <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Scope</th>
                <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Permissions</th>
                <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Created</th>
                <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Expires</th>
                <th className="text-right px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium"></th>
              </tr>
            </thead>
            <tbody>
              {loading && tokens.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-[var(--muted)]">
                    Loading...
                  </td>
                </tr>
              )}
              {!loading && tokens.length === 0 && (
                <tr>
                  <td colSpan={7} className="px-4 py-8 text-center text-[var(--muted)]">
                    No tokens yet. Create one to get started.
                  </td>
                </tr>
              )}
              {tokens.map((t) => (
                <tr key={t.id} className="border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--border)]/20">
                  <td className="px-4 py-3 font-medium">{t.name}</td>
                  <td className="px-4 py-3 font-mono text-xs text-[var(--muted)]">{t.token_prefix}...</td>
                  <td className="px-4 py-3 text-xs text-[var(--muted)]">{t.model_id || "All models"}</td>
                  <td className="px-4 py-3">
                    <div className="flex gap-1">
                      {t.permissions.map((p) => (
                        <span
                          key={p}
                          className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--accent-soft)] text-[var(--accent)]"
                        >
                          {p}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-xs text-[var(--muted)]">
                    {t.created_at ? new Date(t.created_at).toLocaleDateString() : "-"}
                  </td>
                  <td className="px-4 py-3 text-xs text-[var(--muted)]">
                    {t.expires_at ? new Date(t.expires_at).toLocaleDateString() : "Never"}
                  </td>
                  <td className="px-4 py-3 text-right">
                    <button
                      onClick={() => setRevokeTarget({ id: t.id, name: t.name })}
                      className="p-1.5 rounded-lg text-red-400 hover:bg-red-400/10 transition cursor-pointer"
                      title="Revoke token"
                    >
                      <HiOutlineTrash size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Create token modal */}
      {showCreate && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div
            className="w-full max-w-md bg-[var(--bg)] text-[var(--fg)] border border-[var(--border)] rounded-2xl shadow-2xl p-6 space-y-4"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-semibold flex items-center gap-2">
                <HiOutlineKey size={20} className="text-[var(--accent)]" />
                {createdToken ? "Token Created" : "Create API Token"}
              </div>
              <button
                onClick={resetForm}
                className="text-xs text-[var(--muted)] hover:text-[var(--accent)] cursor-pointer"
              >
                Close
              </button>
            </div>

            {createdToken ? (
              <div className="space-y-4">
                <p className="text-sm text-[var(--muted)]">
                  Copy this token now — it won't be shown again.
                </p>
                <div className="flex items-center gap-2">
                  <div className="flex-1 font-mono text-xs bg-[var(--page-bg)] border border-[var(--border)] rounded-xl px-3 py-2 break-all select-all">
                    {createdToken}
                  </div>
                  <button
                    onClick={() => {
                      navigator.clipboard.writeText(createdToken);
                      setCopied(true);
                      setTimeout(() => setCopied(false), 1500);
                    }}
                    className={`p-2 rounded-lg border transition cursor-pointer ${
                      copied
                        ? "border-emerald-400 text-emerald-400"
                        : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--accent)] hover:border-[var(--accent)]"
                    }`}
                    title="Copy"
                  >
                    {copied ? <HiCheck size={16} /> : <HiOutlineDuplicate size={16} />}
                  </button>
                </div>
                <button
                  onClick={resetForm}
                  className="w-full px-4 py-2 text-sm font-semibold rounded-xl bg-[var(--accent)] btn-primary shadow shadow-[var(--accent)]/30 hover:opacity-95 transition cursor-pointer"
                >
                  Done
                </button>
              </div>
            ) : (
              <div className="space-y-3">
                <div className="space-y-1">
                  <label className="text-xs font-medium">Name</label>
                  <input
                    type="text"
                    value={formName}
                    onChange={(e) => setFormName(e.target.value)}
                    placeholder="e.g. boomi-prod, mcp-client"
                    className="w-full px-3 py-2 bg-[var(--page-bg)] border border-[var(--border)] rounded-xl text-sm outline-none focus:border-[var(--accent)]"
                  />
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-medium">Model scope</label>
                  <input
                    type="text"
                    value={formModelId}
                    onChange={(e) => setFormModelId(e.target.value)}
                    placeholder="All models (leave empty)"
                    className="w-full px-3 py-2 bg-[var(--page-bg)] border border-[var(--border)] rounded-xl text-sm outline-none focus:border-[var(--accent)]"
                  />
                  <p className="text-[10px] text-[var(--muted)]">Leave empty for access to all models.</p>
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-medium">Permissions</label>
                  <div className="flex gap-2">
                    {["read", "write", "admin"].map((perm) => (
                      <button
                        key={perm}
                        onClick={() => togglePerm(perm)}
                        className={`px-3 py-1.5 text-xs rounded-xl border transition cursor-pointer ${
                          formPerms.has(perm)
                            ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--accent-soft)]"
                            : "border-[var(--border)] text-[var(--muted)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                        }`}
                      >
                        {perm}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="space-y-1">
                  <label className="text-xs font-medium">Expires</label>
                  <select
                    value={formExpiry}
                    onChange={(e) => setFormExpiry(e.target.value)}
                    className="w-full px-3 py-2 bg-[var(--page-bg)] border border-[var(--border)] rounded-xl text-sm outline-none focus:border-[var(--accent)]"
                  >
                    <option value="30">30 days</option>
                    <option value="90">90 days</option>
                    <option value="365">1 year</option>
                    <option value="never">Never</option>
                  </select>
                </div>

                <div className="flex items-center justify-end gap-2 pt-2">
                  <button
                    onClick={resetForm}
                    className="px-3 py-2 text-xs rounded-xl border border-[var(--border)] text-[var(--muted)] hover:text-[var(--accent)] hover:border-[var(--accent)] cursor-pointer"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleCreate}
                    disabled={!formName.trim() || creating}
                    className="px-4 py-2 text-xs font-semibold rounded-xl bg-[var(--accent)] btn-primary shadow shadow-[var(--accent)]/30 hover:opacity-95 transition disabled:opacity-50 cursor-pointer"
                  >
                    {creating ? "Creating..." : "Create Token"}
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      <ConfirmDialog
        open={revokeTarget !== null}
        title="Revoke Token"
        message={`Revoke token "${revokeTarget?.name}"? Any integrations using this token will immediately lose access. This cannot be undone.`}
        confirmLabel="Revoke Token"
        variant="danger"
        onConfirm={executeRevoke}
        onCancel={() => setRevokeTarget(null)}
      />
    </div>
  );
}
