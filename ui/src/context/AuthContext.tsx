import React, { createContext, useState, useMemo, useContext, useCallback } from "react";
import { ApiClient } from "../lib/apiClient";

interface User {
  id?: string;
  email?: string;
  org_id?: string;
  role?: string;
  first_name?: string;
  last_name?: string;
}

interface AuthContextValue {
  user: User | null;
  token: string | null;
  orgId: string | null;
  api: ApiClient;
  version: string | null;
  login: () => Promise<{ user_code: string; device_code: string; verification_url: string; interval: number }>;
  pollLogin: (deviceCode: string) => Promise<"pending" | "approved" | "expired">;
  logout: () => void;
  checkSession: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

export const AuthProvider = ({ children }: { children: React.ReactNode }) => {
  const [token, setToken] = useState<string | null>(localStorage.getItem("glyphh_token"));
  const [orgId, setOrgId] = useState<string | null>(localStorage.getItem("glyphh_org_id"));
  const [user, setUser] = useState<User | null>(() => {
    const raw = localStorage.getItem("glyphh_user");
    return raw ? JSON.parse(raw) : null;
  });
  const [version, setVersion] = useState<string | null>(null);

  const clearAuth = useCallback(() => {
    setToken(null);
    setOrgId(null);
    setUser(null);
    localStorage.removeItem("glyphh_token");
    localStorage.removeItem("glyphh_platform_token");
    localStorage.removeItem("glyphh_refresh_token");
    localStorage.removeItem("glyphh_org_id");
    localStorage.removeItem("glyphh_user");
  }, []);

  const api = useMemo(
    () =>
      new ApiClient({
        baseUrl: "",
        getToken: () => token,
        onUnauthorized: () => {
          // Try refresh before clearing
          const refreshToken = localStorage.getItem("glyphh_refresh_token");
          if (!refreshToken) {
            clearAuth();
            return;
          }
          fetch("/ui/auth/refresh", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ refresh_token: refreshToken }),
          })
            .then((res) => (res.ok ? res.json() : Promise.reject()))
            .then((data) => {
              const newToken = data.runtime_token || data.access_token;
              if (newToken) {
                setToken(newToken);
                localStorage.setItem("glyphh_token", newToken);
                if (data.access_token) {
                  localStorage.setItem("glyphh_platform_token", data.access_token);
                }
                if (data.refresh_token) {
                  localStorage.setItem("glyphh_refresh_token", data.refresh_token);
                }
              } else {
                clearAuth();
              }
            })
            .catch(() => clearAuth());
        },
      }),
    [token, clearAuth]
  );

  const checkSession = useCallback(async (): Promise<boolean> => {
    try {
      const session = await api.get<{
        authenticated: boolean;
        org_id?: string;
        user_id?: string;
        role?: string;
        version?: string;
      }>("/ui/session");
      if (session.version) setVersion(session.version);
      if (session.authenticated && session.org_id) {
        setOrgId(session.org_id);
        localStorage.setItem("glyphh_org_id", session.org_id);
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, [api]);

  const login = useCallback(async () => {
    const data = await api.post<{
      user_code: string;
      device_code: string;
      verification_url: string;
      interval: number;
    }>("/ui/auth/device/start");
    return data;
  }, [api]);

  const pollLogin = useCallback(
    async (deviceCode: string): Promise<"pending" | "approved" | "expired"> => {
      const data = await api.post<{
        status: string;
        access_token?: string;
        runtime_token?: string;
        refresh_token?: string;
        user?: User;
      }>("/ui/auth/device/poll", { device_code: deviceCode });

      if (data.status === "approved") {
        const newToken = data.runtime_token || data.access_token || null;
        const newUser = data.user || null;
        const newOrgId = newUser?.org_id || null;

        if (newToken) {
          setToken(newToken);
          localStorage.setItem("glyphh_token", newToken);
        }
        if (data.access_token) {
          localStorage.setItem("glyphh_platform_token", data.access_token);
        }
        if (data.refresh_token) {
          localStorage.setItem("glyphh_refresh_token", data.refresh_token);
        }
        if (newUser) {
          setUser(newUser);
          localStorage.setItem("glyphh_user", JSON.stringify(newUser));
        }
        if (newOrgId) {
          setOrgId(newOrgId);
          localStorage.setItem("glyphh_org_id", newOrgId);
        }
        return "approved";
      }

      if (data.status === "expired") return "expired";
      return "pending";
    },
    [api]
  );

  const logout = useCallback(() => {
    clearAuth();
  }, [clearAuth]);

  return (
    <AuthContext.Provider
      value={{ user, token, orgId, api, version, login, pollLogin, logout, checkSession }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be inside AuthProvider");
  return ctx;
};
