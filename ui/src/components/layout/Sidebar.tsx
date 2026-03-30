import { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";
import { useTheme } from "../../context/ThemeContext";
import { useAuth } from "../../context/AuthContext";
import {
  HiOutlineCubeTransparent,
  HiOutlineKey,
  HiOutlineUserCircle,
  HiOutlineLockClosed,
  HiOutlineMoon,
} from "react-icons/hi";

export default function Sidebar() {
  const { theme, preference, setPreference } = useTheme();
  const { user, logout, version } = useAuth();
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    if (!showSettings) return;
    const onClick = (e: MouseEvent) => {
      const target = e.target as HTMLElement | null;
      if (target && target.closest("[data-settings-modal]")) return;
      setShowSettings(false);
    };
    document.addEventListener("mousedown", onClick);
    return () => document.removeEventListener("mousedown", onClick);
  }, [showSettings]);

  const navBase =
    "flex flex-col items-center gap-1 px-2 py-1 rounded-2xl text-sm transition text-[var(--muted)]";
  const iconWrapper =
    "flex items-center justify-center w-11 h-11 rounded-full border border-transparent transition";
  const iconActive = "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]";

  return (
    <aside className="w-[64px] h-screen bg-[var(--sidebar)] border-r border-[var(--sidebar-border)] flex flex-col justify-between py-6">
      <div className="flex flex-col items-center gap-8">
        <div className="flex flex-col items-center gap-1">
          <img src="/glyphh-logo-square.png" alt="Glyphh" className="w-10 h-10" />
        </div>
        <nav className="flex flex-col items-center gap-2">
          {[
            { to: "/models", icon: HiOutlineCubeTransparent, label: "Models" },
            { to: "/tokens", icon: HiOutlineKey, label: "Tokens" },
          ].map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                `${navBase} ${isActive ? "text-[var(--accent)]" : "text-[var(--muted)]"}`
              }
              title={label}
            >
              {({ isActive }) => (
                <>
                  <div
                    className={`${iconWrapper} ${
                      isActive
                        ? iconActive
                        : "border-[var(--border)] text-[var(--muted)] hover:border-[var(--accent)] hover:text-[var(--accent)]"
                    }`}
                  >
                    <Icon size={22} />
                  </div>
                  <span className="text-[10px]">{label}</span>
                </>
              )}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="flex flex-col items-center px-1">
        <button
          type="button"
          onClick={() => setShowSettings(true)}
          className="flex items-center justify-center w-10 h-10 rounded-full hover:border-[var(--accent)] cursor-pointer"
          title="Settings"
        >
          <HiOutlineUserCircle size={30} className="text-[var(--muted)]" />
        </button>
      </div>

      {showSettings && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div
            data-settings-modal
            className="w-full max-w-md bg-[var(--bg)] text-[var(--fg)] border border-[var(--border)] rounded-2xl shadow-2xl p-6 space-y-4"
          >
            <div className="flex items-center justify-between mb-2">
              <div className="text-lg font-semibold flex items-center gap-2">
                <HiOutlineUserCircle size={24} className="text-[var(--accent)]" /> Settings
              </div>
              <button
                onClick={() => setShowSettings(false)}
                className="text-xs text-[var(--muted)] hover:text-[var(--accent)] cursor-pointer"
              >
                Close
              </button>
            </div>

            {user && (
              <div className="text-sm text-[var(--muted)] border-b border-[var(--border)] pb-3">
                <div className="font-medium text-[var(--fg)]">
                  {user.first_name} {user.last_name}
                </div>
                <div className="text-xs">{user.email}</div>
              </div>
            )}

            <div className="space-y-3">
              <div>
                <div className="text-xs font-semibold text-[var(--muted)] uppercase mb-2 flex items-center gap-1.5">
                  <HiOutlineMoon size={14} /> Theme
                </div>
                <div className="flex gap-2">
                  {(["light", "dark", "system"] as const).map((opt) => (
                    <button
                      key={opt}
                      onClick={() => setPreference(opt)}
                      className={`px-3 py-2 text-xs rounded-xl border transition cursor-pointer ${
                        preference === opt
                          ? "border-[var(--accent)] text-[var(--accent)]"
                          : "border-[var(--border)] text-[var(--muted)] hover:text-[var(--accent)] hover:border-[var(--accent)]"
                      }`}
                    >
                      {opt.charAt(0).toUpperCase() + opt.slice(1)}
                    </button>
                  ))}
                </div>
                <div className="text-[11px] text-[var(--muted)] mt-1">
                  Current: {theme} ({preference === "system" ? "system" : "manual"})
                </div>
              </div>

              {version && (
                <div className="text-xs text-[var(--muted)]">
                  Runtime v{version}
                </div>
              )}

              <button
                onClick={() => {
                  setShowSettings(false);
                  logout();
                }}
                className="flex items-center gap-2 px-3 py-2 text-xs font-semibold rounded-xl border border-[var(--border)] text-[var(--muted)] hover:text-[var(--fg)] transition w-full cursor-pointer"
              >
                <HiOutlineLockClosed size={16} />
                Log out
              </button>
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
