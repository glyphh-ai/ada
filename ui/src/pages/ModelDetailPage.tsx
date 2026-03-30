import { useEffect, useState, useRef, useCallback } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import PageHeader from "../components/layout/PageHeader";
import ProgressBar from "../components/ProgressBar";
import ConfirmDialog from "../components/ConfirmDialog";
import { HiOutlineCubeTransparent, HiOutlinePencil, HiOutlineX } from "react-icons/hi";

type ReadyResponse = { ready: boolean; status?: string; meta_name?: string };

type GlyphRow = {
  id: string;
  concept_text?: string;
  node_type?: string;
  record_type?: string;
  has_embedding?: boolean;
  vector_dim?: number;
  created_at?: string;
};

export default function ModelDetailPage() {
  const { modelId } = useParams<{ modelId: string }>();
  const { api, orgId } = useAuth();
  const navigate = useNavigate();
  const [name, setName] = useState(modelId || "");
  const [status, setStatus] = useState("-");
  const [glyphs, setGlyphs] = useState(0);
  const [vectors, setVectors] = useState(0);
  const [data, setData] = useState<GlyphRow[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(0);
  const [loadingData, setLoadingData] = useState(false);
  type LoadPhase = "idle" | "parsing" | "submitting" | "processing" | "finalizing" | "done" | "error";
  const [loadPhase, setLoadPhase] = useState<LoadPhase>("idle");
  const [loadLabel, setLoadLabel] = useState("");
  const [loadFraction, setLoadFraction] = useState(0);
  const [reEncoding, setReEncoding] = useState(false);
  const loadPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [confirmAction, setConfirmAction] = useState<"clear" | "delete" | null>(null);
  const [selectedGlyph, setSelectedGlyph] = useState<GlyphRow | null>(null);
  const [editText, setEditText] = useState("");
  const [saving, setSaving] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const LIMIT = 20;

  const totalPages = Math.max(1, Math.ceil(total / LIMIT));
  const offset = page * LIMIT;

  const loadStatus = useCallback(async () => {
    if (!orgId || !modelId) return null;
    try {
      const [ready, counts] = await Promise.all([
        api.get<ReadyResponse>(`/${orgId}/${modelId}/ready`).catch(() => null),
        api.get<{ glyphs: number; vectors: number }>(`/${orgId}/${modelId}/data/count`).catch(() => null),
      ]);
      if (ready) {
        setName(ready.meta_name || modelId);
        const s = ready.ready ? "Ready" : (ready.status || "Unknown");
        setStatus(s);
      }
      if (counts) {
        setGlyphs(counts.glyphs ?? 0);
        setVectors(counts.vectors ?? 0);
      }
      return ready?.status ?? null;
    } catch {
      return null;
    }
  }, [orgId, modelId, api]);

  const loadData = useCallback(async (newPage = page) => {
    if (!orgId || !modelId) return;
    setLoadingData(true);
    try {
      const res = await api.get<{ glyphs: GlyphRow[]; total: number }>(
        `/${orgId}/${modelId}/data?limit=${LIMIT}&offset=${newPage * LIMIT}`
      );
      setData(res.glyphs || []);
      setTotal(res.total || 0);
      setPage(newPage);
    } catch {
      setData([]);
    } finally {
      setLoadingData(false);
    }
  }, [orgId, modelId, api, page]);

  const encodingPollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    loadStatus();
    loadData(0);
  }, [orgId, modelId]);

  // Poll while model is encoding or locked — refresh counts so user sees progress
  useEffect(() => {
    const isBusy = status === "encoding" || status === "locked";
    if (isBusy && !encodingPollRef.current) {
      encodingPollRef.current = setInterval(async () => {
        const s = await loadStatus();
        if (s !== "encoding" && s !== "locked") {
          if (encodingPollRef.current) {
            clearInterval(encodingPollRef.current);
            encodingPollRef.current = null;
          }
          loadData(0);
        }
      }, 2000);
    } else if (!isBusy && encodingPollRef.current) {
      clearInterval(encodingPollRef.current);
      encodingPollRef.current = null;
    }
    return () => {
      if (encodingPollRef.current) {
        clearInterval(encodingPollRef.current);
        encodingPollRef.current = null;
      }
    };
  }, [status, loadStatus]);

  const handleReEncode = async () => {
    if (!orgId || !modelId) return;
    setReEncoding(true);
    try {
      await api.post(`/${orgId}/${modelId}/model/re-encode`);
      const poll = setInterval(async () => {
        try {
          const ready = await api.get<{ ready: boolean; status?: string }>(`/${orgId}/${modelId}/ready`);
          if (ready.status !== "locked") {
            clearInterval(poll);
            setReEncoding(false);
            await loadStatus();
            await loadData(0);
          }
        } catch {
          clearInterval(poll);
          setReEncoding(false);
        }
      }, 2000);
    } catch (e: unknown) {
      setReEncoding(false);
      alert("Re-encode failed: " + (e instanceof Error ? e.message : ""));
    }
  };

  const executeClear = useCallback(async () => {
    if (!orgId || !modelId) return;
    setConfirmAction(null);
    try {
      await api.delete(`/${orgId}/${modelId}/data`);
      await loadStatus();
      await loadData(0);
    } catch (e: unknown) {
      alert("Clear failed: " + (e instanceof Error ? e.message : ""));
    }
  }, [orgId, modelId, api]);

  const executeDelete = useCallback(async () => {
    if (!orgId || !modelId) return;
    setConfirmAction(null);
    try {
      await api.delete(`/${orgId}/${modelId}/model`);
      navigate("/models", { replace: true });
    } catch (e: unknown) {
      alert("Delete failed: " + (e instanceof Error ? e.message : ""));
    }
  }, [orgId, modelId, api, navigate]);

  const openGlyph = (g: GlyphRow) => {
    setSelectedGlyph(g);
    setEditText(g.concept_text || "");
  };

  const saveGlyph = async () => {
    if (!orgId || !modelId || !selectedGlyph) return;
    setSaving(true);
    try {
      await api.put(`/${orgId}/${modelId}/data/${selectedGlyph.id}`, {
        concept_text: editText,
      });
      setSelectedGlyph(null);
      await loadData(page);
    } catch (e: unknown) {
      alert("Save failed: " + (e instanceof Error ? e.message : ""));
    } finally {
      setSaving(false);
    }
  };

  const clearLoadPoll = () => {
    if (loadPollRef.current) {
      clearInterval(loadPollRef.current);
      loadPollRef.current = null;
    }
  };

  const startFinalizePoll = () => {
    // After job completes, poll readiness until model is no longer encoding
    setLoadPhase("finalizing");
    setLoadLabel("Indexing vectors...");

    loadPollRef.current = setInterval(async () => {
      try {
        const s = await loadStatus();
        if (s !== "encoding" && s !== "locked") {
          clearLoadPoll();
          setLoadPhase("done");
          setLoadFraction(1);
          setLoadLabel("Complete");
          await loadData(0);
          setTimeout(() => setLoadPhase("idle"), 2000);
        }
      } catch {
        clearLoadPoll();
        setLoadPhase("idle");
      }
    }, 2000);
  };

  const handleDataFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !orgId || !modelId) return;
    e.target.value = "";

    // Phase 1: Parsing
    setLoadPhase("parsing");
    setLoadFraction(0);
    setLoadLabel("Parsing file...");

    try {
      const text = await file.text();
      let records: unknown[];
      const trimmed = text.trim();
      try {
        const parsed = JSON.parse(trimmed);
        if (Array.isArray(parsed)) records = parsed;
        else if (parsed.records && Array.isArray(parsed.records)) records = parsed.records;
        else throw new Error("Expected array or {records: [...]}");
      } catch {
        records = trimmed.split("\n").filter(Boolean).map((l) => JSON.parse(l));
      }

      // Phase 2: Submitting
      setLoadPhase("submitting");
      setLoadLabel(`Submitting ${records.length.toLocaleString()} records...`);

      const res = await api.post<{ job_id?: string }>(`/${orgId}/${modelId}/listener`, {
        records,
        batch_size: 50,
      });

      if (res.job_id) {
        // Phase 3: Processing — poll job
        setLoadPhase("processing");
        setLoadFraction(0);
        setLoadLabel(`0 / ${records.length} records`);

        loadPollRef.current = setInterval(async () => {
          try {
            const job = await api.get<{
              status: string;
              total: number;
              processed: number;
              encoded: number;
              progress: number;
              error?: string;
            }>(`/${orgId}/${modelId}/listener/jobs/${res.job_id}`);

            setLoadFraction((job.progress ?? 0) / 100);
            setLoadLabel(`${job.encoded ?? job.processed} / ${job.total} records`);

            if (job.status === "completed" || job.status === "done") {
              clearLoadPoll();
              startFinalizePoll();
            } else if (job.status === "error" || job.status === "failed") {
              clearLoadPoll();
              setLoadPhase("error");
              setLoadLabel(`Error: ${job.error || "Unknown"}`);
            }
          } catch {
            clearLoadPoll();
            setLoadPhase("error");
            setLoadLabel("Connection lost during processing");
          }
        }, 1000);
      } else {
        // No job ID — synchronous load, go straight to finalize
        startFinalizePoll();
      }
    } catch (err: unknown) {
      setLoadPhase("error");
      setLoadLabel("Error: " + (err instanceof Error ? err.message : ""));
    }
  };

  // Build page numbers: show up to 7 pages with ellipsis
  const pageNumbers = (): (number | "...")[] => {
    if (totalPages <= 7) return Array.from({ length: totalPages }, (_, i) => i);
    const pages: (number | "...")[] = [];
    if (page <= 3) {
      for (let i = 0; i < 5; i++) pages.push(i);
      pages.push("...", totalPages - 1);
    } else if (page >= totalPages - 4) {
      pages.push(0, "...");
      for (let i = totalPages - 5; i < totalPages; i++) pages.push(i);
    } else {
      pages.push(0, "...", page - 1, page, page + 1, "...", totalPages - 1);
    }
    return pages;
  };

  return (
    <div className="space-y-4 pb-12">
      <PageHeader
        title={
          <span className="inline-flex items-center gap-2">
            <HiOutlineCubeTransparent size={22} className="text-[var(--accent)]" />
            <span>{name}</span>
          </span>
        }
        subtitle={modelId}
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={handleReEncode}
              className="px-3 py-2 text-xs rounded-xl border border-[var(--border)] text-[var(--muted)] hover:text-[var(--accent)] hover:border-[var(--accent)] cursor-pointer"
            >
              Re-encode
            </button>
            <button
              onClick={() => setConfirmAction("clear")}
              className="px-3 py-2 text-xs rounded-xl border border-red-500/30 text-red-400 hover:border-red-400 cursor-pointer"
            >
              Clear Data
            </button>
            <button
              onClick={() => setConfirmAction("delete")}
              className="px-3 py-2 text-xs rounded-xl border border-red-500/30 text-red-400 hover:border-red-400 cursor-pointer"
            >
              Delete
            </button>
          </div>
        }
      />

      <div className="px-4 space-y-6">
        {/* Back link */}
        <button
          onClick={() => navigate("/models")}
          className="text-sm text-[var(--muted)] hover:text-[var(--fg)] cursor-pointer"
        >
          &larr; Back to models
        </button>

        {/* Stats */}
        <div className="flex gap-4">
          {[
            { label: "Glyphs", value: glyphs.toLocaleString() },
            { label: "Vectors", value: vectors.toLocaleString() },
            { label: "Status", value: status },
          ].map((s) => (
            <div
              key={s.label}
              className="p-4 rounded-xl border border-[var(--border)] bg-[var(--card-bg)] min-w-[140px]"
            >
              <div className="text-[11px] text-[var(--muted)] uppercase tracking-wide">
                {s.label}
              </div>
              <div className="text-xl font-semibold mt-1">{s.value}</div>
            </div>
          ))}
        </div>

        {/* Drop zone */}
        <div
          onClick={() => loadPhase === "idle" && fileRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); e.currentTarget.classList.add("border-[var(--accent)]", "bg-[var(--accent-soft)]"); }}
          onDragLeave={(e) => { e.currentTarget.classList.remove("border-[var(--accent)]", "bg-[var(--accent-soft)]"); }}
          onDrop={(e) => {
            e.preventDefault();
            e.currentTarget.classList.remove("border-[var(--accent)]", "bg-[var(--accent-soft)]");
            const file = e.dataTransfer.files[0];
            if (file) {
              const input = fileRef.current;
              if (input) {
                const dt = new DataTransfer();
                dt.items.add(file);
                input.files = dt.files;
                input.dispatchEvent(new Event("change", { bubbles: true }));
              }
            }
          }}
          className="border-2 border-dashed border-[var(--border)] rounded-xl p-10 text-center text-[var(--muted)] hover:border-[var(--accent)] hover:bg-[var(--accent-soft)] transition cursor-pointer"
        >
          <input
            ref={fileRef}
            type="file"
            accept=".json,.jsonl"
            className="hidden"
            onChange={handleDataFile}
          />
          <div className="text-2xl mb-2">&#128230;</div>
          <p className="text-sm">Drop a concepts.json file here to load data</p>
          <p className="text-xs text-[var(--muted)] mt-1">JSON array, {`{records: [...]}`}, or JSONL</p>
        </div>

        {(reEncoding || status === "encoding" || status === "locked") && (
          <div className="p-3 rounded-xl border border-[var(--accent)]/30 bg-[var(--card-bg)]">
            <ProgressBar
              progress={glyphs > 0 && vectors >= 0 ? Math.min(vectors / Math.max(glyphs, 1), 0.99) : 0}
              indeterminate={vectors === 0 && glyphs === 0}
              label={
                status === "encoding"
                  ? `Encoding exemplars${glyphs > 0 ? ` — ${vectors.toLocaleString()} / ${glyphs.toLocaleString()} vectors` : "..."}`
                  : `Re-encoding vectors${glyphs > 0 ? ` — ${vectors.toLocaleString()} / ${glyphs.toLocaleString()}` : "..."}`
              }
            />
          </div>
        )}

        {loadPhase !== "idle" && (
          <div className="p-3 rounded-xl border border-[var(--border)] bg-[var(--card-bg)]">
            {loadPhase === "error" ? (
              <div className="flex items-center justify-between">
                <div className="font-mono text-xs text-red-400">{loadLabel}</div>
                <button
                  onClick={() => setLoadPhase("idle")}
                  className="text-xs text-[var(--muted)] hover:text-[var(--fg)] cursor-pointer"
                >
                  Dismiss
                </button>
              </div>
            ) : loadPhase === "done" ? (
              <ProgressBar progress={1} label={loadLabel} />
            ) : loadPhase === "processing" ? (
              <ProgressBar progress={loadFraction} label={loadLabel} />
            ) : (
              <ProgressBar progress={0} indeterminate label={loadLabel} />
            )}
          </div>
        )}

        {/* Data table */}
        <div>
          <h2 className="text-sm font-semibold mb-3">Data</h2>

          <div className="border border-[var(--border)] rounded-xl overflow-hidden bg-[var(--card-bg)]">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">ID</th>
                  <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Concept</th>
                  <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Type</th>
                  <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Vector</th>
                  <th className="text-left px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Created</th>
                  <th className="text-right px-4 py-2 text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium"></th>
                </tr>
              </thead>
              <tbody>
                {loadingData && data.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-[var(--muted)]">
                      Loading...
                    </td>
                  </tr>
                )}
                {!loadingData && data.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-[var(--muted)]">
                      No data loaded yet
                    </td>
                  </tr>
                )}
                {data.map((g) => (
                  <tr
                    key={g.id}
                    className="border-b border-[var(--border)] last:border-b-0 hover:bg-[var(--border)]/20 cursor-pointer"
                    onClick={() => openGlyph(g)}
                  >
                    <td className="px-4 py-2 font-mono text-[11px] text-[var(--muted)]">
                      {String(g.id).slice(0, 8)}
                    </td>
                    <td className="px-4 py-2 max-w-[300px] truncate">
                      {(g.concept_text || "").slice(0, 80)}
                    </td>
                    <td className="px-4 py-2">
                      <div className="flex items-center gap-1.5">
                        {g.node_type && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--accent-soft)] text-[var(--accent)]">
                            {g.node_type}
                          </span>
                        )}
                        {g.record_type && g.record_type !== "data" && (
                          <span className="text-[10px] px-2 py-0.5 rounded-full bg-[var(--border)]/40 text-[var(--muted)]">
                            {g.record_type}
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-2 font-mono text-[11px] text-[var(--muted)]">
                      {g.has_embedding ? `${g.vector_dim}d` : "—"}
                    </td>
                    <td className="px-4 py-2 text-xs text-[var(--muted)]">
                      {g.created_at ? new Date(g.created_at).toLocaleDateString() : "—"}
                    </td>
                    <td className="px-4 py-2 text-right">
                      <button
                        onClick={(e) => { e.stopPropagation(); openGlyph(g); }}
                        className="p-1 rounded-lg text-[var(--muted)] hover:text-[var(--accent)] transition"
                        title="View / Edit"
                      >
                        <HiOutlinePencil size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {total > 0 && (
            <div className="flex items-center justify-between mt-4">
              <div className="text-xs text-[var(--muted)]">
                {offset + 1}–{Math.min(offset + LIMIT, total)} of {total}
              </div>
              <div className="flex items-center gap-1">
                <button
                  onClick={() => loadData(page - 1)}
                  disabled={page === 0}
                  className="px-2.5 py-1 text-xs rounded-lg border border-[var(--border)] text-[var(--muted)] hover:text-[var(--accent)] hover:border-[var(--accent)] transition cursor-pointer disabled:opacity-30 disabled:cursor-default disabled:hover:text-[var(--muted)] disabled:hover:border-[var(--border)]"
                >
                  &lsaquo; Prev
                </button>
                {pageNumbers().map((p, i) =>
                  p === "..." ? (
                    <span key={`e${i}`} className="px-1.5 text-xs text-[var(--muted)]">
                      &hellip;
                    </span>
                  ) : (
                    <button
                      key={p}
                      onClick={() => loadData(p)}
                      className={`px-2.5 py-1 text-xs rounded-lg border transition cursor-pointer ${
                        p === page
                          ? "border-[var(--accent)] text-[var(--accent)] bg-[var(--accent-soft)]"
                          : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--accent)] hover:border-[var(--accent)]"
                      }`}
                    >
                      {p + 1}
                    </button>
                  ),
                )}
                <button
                  onClick={() => loadData(page + 1)}
                  disabled={page >= totalPages - 1}
                  className="px-2.5 py-1 text-xs rounded-lg border border-[var(--border)] text-[var(--muted)] hover:text-[var(--accent)] hover:border-[var(--accent)] transition cursor-pointer disabled:opacity-30 disabled:cursor-default disabled:hover:text-[var(--muted)] disabled:hover:border-[var(--border)]"
                >
                  Next &rsaquo;
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Glyph detail / edit drawer */}
      {selectedGlyph && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50" onClick={() => setSelectedGlyph(null)}>
          <div
            className="w-full max-w-lg bg-[var(--bg)] text-[var(--fg)] border border-[var(--border)] rounded-2xl shadow-2xl p-6 space-y-4 max-h-[80vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between">
              <div className="text-base font-semibold">Glyph Detail</div>
              <button
                onClick={() => setSelectedGlyph(null)}
                className="p-1 rounded-lg text-[var(--muted)] hover:text-[var(--accent)] cursor-pointer"
              >
                <HiOutlineX size={18} />
              </button>
            </div>

            {/* ID */}
            <div className="space-y-1">
              <label className="text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">ID</label>
              <div className="font-mono text-xs text-[var(--muted)] bg-[var(--page-bg)] border border-[var(--border)] rounded-xl px-3 py-2 select-all">
                {selectedGlyph.id}
              </div>
            </div>

            {/* Concept text (editable) */}
            <div className="space-y-1">
              <label className="text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Concept Text</label>
              <textarea
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                rows={4}
                className="w-full px-3 py-2 bg-[var(--page-bg)] border border-[var(--border)] rounded-xl text-sm outline-none focus:border-[var(--accent)] resize-y font-mono"
              />
            </div>

            {/* Metadata fields */}
            <div className="grid grid-cols-2 gap-3">
              <div className="space-y-1">
                <label className="text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Node Type</label>
                <div className="text-sm px-3 py-2 bg-[var(--page-bg)] border border-[var(--border)] rounded-xl">
                  {selectedGlyph.node_type || "—"}
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Record Type</label>
                <div className="text-sm px-3 py-2 bg-[var(--page-bg)] border border-[var(--border)] rounded-xl">
                  {selectedGlyph.record_type || "—"}
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Vector</label>
                <div className="text-sm px-3 py-2 bg-[var(--page-bg)] border border-[var(--border)] rounded-xl">
                  {selectedGlyph.has_embedding ? `${selectedGlyph.vector_dim}d` : "No embedding"}
                </div>
              </div>
              <div className="space-y-1">
                <label className="text-[11px] text-[var(--muted)] uppercase tracking-wide font-medium">Created</label>
                <div className="text-sm px-3 py-2 bg-[var(--page-bg)] border border-[var(--border)] rounded-xl">
                  {selectedGlyph.created_at ? new Date(selectedGlyph.created_at).toLocaleString() : "—"}
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex items-center justify-end gap-2 pt-2">
              <button
                onClick={() => setSelectedGlyph(null)}
                className="px-3 py-2 text-xs rounded-xl border border-[var(--border)] text-[var(--muted)] hover:text-[var(--fg)] transition cursor-pointer"
              >
                Cancel
              </button>
              <button
                onClick={saveGlyph}
                disabled={saving || editText === (selectedGlyph.concept_text || "")}
                className="px-4 py-2 text-xs font-semibold rounded-xl bg-[var(--accent)] btn-primary shadow shadow-[var(--accent)]/30 hover:opacity-95 transition disabled:opacity-50 cursor-pointer"
              >
                {saving ? "Saving..." : "Save Changes"}
              </button>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={confirmAction === "clear"}
        title="Clear Model Data"
        message={`This will permanently delete all glyphs and vectors for ${modelId}. The model will remain deployed but empty.`}
        confirmLabel="Clear All Data"
        variant="danger"
        onConfirm={executeClear}
        onCancel={() => setConfirmAction(null)}
      />

      <ConfirmDialog
        open={confirmAction === "delete"}
        title="Delete Model"
        message={`This will permanently delete ${modelId} and all its data — glyphs, vectors, edges, and configuration. This cannot be undone.`}
        confirmLabel="Delete Model"
        variant="danger"
        onConfirm={executeDelete}
        onCancel={() => setConfirmAction(null)}
      />
    </div>
  );
}
