import { useEffect, useState, useRef, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";
import PageHeader from "../components/layout/PageHeader";
import ProgressBar from "../components/ProgressBar";
import { HiOutlineCubeTransparent } from "react-icons/hi";

type Model = {
  model_id: string;
  name?: string;
  version?: string;
  status?: string;
  glyphs?: number;
};

export default function ModelsPage() {
  const { api, orgId } = useAuth();
  const navigate = useNavigate();
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [deploying, setDeploying] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const loadModels = useCallback(async () => {
    if (!orgId) return;
    try {
      const data = await api.get<{ models: Model[] }>(`/${orgId}/models`);
      setModels(data.models || []);
      return data.models || [];
    } catch {
      setModels([]);
      return [];
    }
  }, [orgId, api]);

  // Initial load
  useEffect(() => {
    setLoading(true);
    loadModels().finally(() => setLoading(false));
  }, [loadModels]);

  // Poll while any model is encoding or deploying
  useEffect(() => {
    const hasEncoding = models.some((m) => m.status === "encoding" || m.status === "locked");
    const shouldPoll = hasEncoding || deploying !== null;

    if (shouldPoll && !pollRef.current) {
      pollRef.current = setInterval(() => {
        loadModels();
      }, 2000);
    } else if (!shouldPoll && pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [models, deploying, loadModels]);

  const handleDeploy = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !orgId) return;
    e.target.value = "";

    const modelId = file.name.replace(/\.glyphh$/, "");
    setDeploying(modelId);
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      await api.postFormData(`/${orgId}/${modelId}/model/deploy`, form);
      await loadModels();
    } catch (err: unknown) {
      alert("Deploy failed: " + (err instanceof Error ? err.message : "Unknown error"));
    } finally {
      setDeploying(null);
    }
  };

  const isModelBusy = (m: Model) =>
    deploying === m.model_id || m.status === "encoding" || m.status === "locked";

  const busyLabel = (m: Model) => {
    if (deploying === m.model_id) return "Deploying...";
    if (m.status === "encoding") return "Encoding exemplars...";
    if (m.status === "locked") return "Re-encoding...";
    return "";
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title={
          <span className="inline-flex items-center gap-2">
            <HiOutlineCubeTransparent size={22} className="text-[var(--accent)]" />
            <span>Models</span>
          </span>
        }
        subtitle="Deployed models on this runtime."
        actions={
          <div className="flex items-center gap-2">
            <input
              ref={fileRef}
              type="file"
              accept=".glyphh"
              className="hidden"
              onChange={handleDeploy}
            />
            <button
              onClick={() => fileRef.current?.click()}
              className="px-3 py-2 text-sm font-semibold rounded-xl bg-[var(--accent)] btn-primary shadow shadow-[var(--accent)]/30 hover:opacity-95 transition cursor-pointer"
            >
              Deploy Model
            </button>
          </div>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4 px-4">
        {loading && models.length === 0 && (
          <div className="col-span-full text-sm text-[var(--muted)]">Loading...</div>
        )}

        {deploying && !models.some((m) => m.model_id === deploying) && (
          <div className="p-4 border border-[var(--accent)]/40 rounded-xl bg-[var(--card-bg)] border-[var(--card-border)]">
            <div className="font-semibold text-base mb-1">{deploying}</div>
            <div className="text-xs text-[var(--muted)] font-mono mb-3">{deploying}</div>
            <ProgressBar progress={0} indeterminate label="Deploying..." />
          </div>
        )}

        {models.map((m) => (
          <div
            key={m.model_id}
            onClick={() => navigate(`/models/${m.model_id}`)}
            className="p-4 border border-[var(--border)] rounded-xl hover:border-[var(--accent)] transition cursor-pointer bg-[var(--card-bg)] border-[var(--card-border)]"
          >
            <div className="flex items-start justify-between mb-2">
              <div className="font-semibold text-base">{m.name || m.model_id}</div>
              <span className={`text-xs px-2 py-1 rounded-full ${
                isModelBusy(m)
                  ? "bg-amber-500/10 text-amber-400"
                  : "bg-[var(--accent-soft)] text-[var(--accent)]"
              }`}>
                {isModelBusy(m) ? busyLabel(m).replace("...", "") : (m.status || "unknown")}
                {!isModelBusy(m) && m.version ? ` \u2022 v${m.version}` : ""}
              </span>
            </div>
            <div className="text-xs text-[var(--muted)] font-mono mb-3">{m.model_id}</div>
            {isModelBusy(m) ? (
              <ProgressBar progress={0} indeterminate label={busyLabel(m)} />
            ) : (
              <div className="grid grid-cols-2 gap-2 text-[11px]">
                <div className="p-2 rounded-lg bg-[var(--border)]/20 border border-[var(--border)]/60">
                  <div className="text-[var(--muted)]">Glyphs</div>
                  <div className="font-semibold text-sm">{m.glyphs ?? 0}</div>
                </div>
                <div className="p-2 rounded-lg bg-[var(--border)]/20 border border-[var(--border)]/60">
                  <div className="text-[var(--muted)]">Status</div>
                  <div className="font-semibold text-sm">{m.status || "unknown"}</div>
                </div>
              </div>
            )}
          </div>
        ))}

        {!loading && models.length === 0 && !deploying && (
          <div className="col-span-full text-center py-16">
            <p className="text-sm text-[var(--muted)] mb-4">No models deployed yet.</p>
            <button
              onClick={() => fileRef.current?.click()}
              className="px-4 py-2 text-sm font-semibold rounded-xl bg-[var(--accent)] btn-primary shadow shadow-[var(--accent)]/30 hover:opacity-95 transition cursor-pointer"
            >
              Deploy your first model
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
