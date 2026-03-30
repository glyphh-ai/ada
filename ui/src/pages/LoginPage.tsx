import { useState, useEffect, useRef } from "react";
import { useAuth } from "../context/AuthContext";
import { useTheme } from "../context/ThemeContext";
import { useNavigate } from "react-router-dom";

export default function LoginPage() {
  const { login, pollLogin, checkSession, version } = useAuth();
  const { theme } = useTheme();
  const navigate = useNavigate();
  const [stage, setStage] = useState<"idle" | "polling" | "error">("idle");
  const [userCode, setUserCode] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    checkSession().then((ok) => {
      if (ok) navigate("/models", { replace: true });
    });
  }, [checkSession, navigate]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startLogin = async () => {
    setStage("polling");
    setErrorMsg("");

    try {
      const data = await login();
      setUserCode(data.user_code);
      window.open(data.verification_url, "_blank");

      timerRef.current = setInterval(async () => {
        try {
          const status = await pollLogin(data.device_code);
          if (status === "approved") {
            if (timerRef.current) clearInterval(timerRef.current);
            navigate("/models", { replace: true });
          } else if (status === "expired") {
            if (timerRef.current) clearInterval(timerRef.current);
            setStage("error");
            setErrorMsg("Code expired. Please try again.");
          }
        } catch {
          // Keep polling on network errors
        }
      }, (data.interval || 5) * 1000);
    } catch (e: unknown) {
      setStage("error");
      setErrorMsg(e instanceof Error ? e.message : "Could not start login.");
    }
  };

  const reset = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setStage("idle");
    setUserCode("");
    setErrorMsg("");
  };

  return (
    <div className="flex flex-col items-center justify-center min-h-screen gap-8">
      <div className="flex flex-col items-center gap-2">
        <img src={theme === "light" ? "/glyphh-logo-dark.png" : "/glyphh-logo.png"} alt="Glyphh" className="h-12" />
      </div>

      <div className="w-full max-w-sm bg-[var(--card-bg)] border border-[var(--card-border)] rounded-2xl p-8 text-center">
        {stage === "idle" && (
          <>
            <h2 className="text-lg font-semibold mb-2">
              Glyphh Runtime{version ? ` v${version}` : ""}
            </h2>
            <p className="text-sm text-[var(--muted)] mb-6">
              Sign in to manage your models and data.
            </p>
            <button
              onClick={startLogin}
              className="w-full px-4 py-3 text-sm font-semibold rounded-xl bg-[var(--accent)] btn-primary shadow shadow-[var(--accent)]/30 hover:opacity-95 transition cursor-pointer"
            >
              Sign In
            </button>
          </>
        )}

        {stage === "polling" && (
          <>
            <h2 className="text-lg font-semibold mb-2">Authorize this device</h2>
            <p className="text-sm text-[var(--muted)] mb-4">
              Enter this code in the browser window that just opened:
            </p>
            <div className="font-mono text-3xl font-bold text-[var(--accent)] tracking-widest my-6">
              {userCode}
            </div>
            <div className="flex items-center justify-center gap-2 text-sm text-[var(--muted)]">
              <span className="inline-block w-4 h-4 border-2 border-[var(--border)] border-t-[var(--accent)] rounded-full animate-spin" />
              Waiting for approval...
            </div>
          </>
        )}

        {stage === "error" && (
          <>
            <p className="text-sm text-red-400 mb-4">{errorMsg}</p>
            <button
              onClick={reset}
              className="px-4 py-2 text-sm rounded-xl border border-[var(--border)] text-[var(--muted)] hover:text-[var(--accent)] hover:border-[var(--accent)] transition cursor-pointer"
            >
              Try Again
            </button>
          </>
        )}
      </div>
    </div>
  );
}
