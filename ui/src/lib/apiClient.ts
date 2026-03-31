export class ApiClient {
  constructor(
    private opts: {
      baseUrl: string;
      getToken: () => string | null;
      onUnauthorized?: () => void;
    }
  ) {}

  get baseUrl() {
    return this.opts.baseUrl;
  }

  getToken() {
    return this.opts.getToken();
  }

  private headers(extra?: HeadersInit, body?: BodyInit): HeadersInit {
    const token = this.opts.getToken();
    const headerEntries: HeadersInit = {
      ...(extra || {}),
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    };
    if (!(body instanceof FormData)) {
      return {
        "Content-Type": "application/json",
        ...headerEntries,
      };
    }
    return headerEntries;
  }

  async request<T>(path: string, init: RequestInit = {}) {
    const res = await fetch(this.opts.baseUrl + path, {
      ...init,
      headers: this.headers(init.headers as HeadersInit, init.body ?? undefined),
    });

    if (res.status === 401 && this.opts.onUnauthorized) {
      this.opts.onUnauthorized();
    }

    if (!res.ok) {
      const text = await res.text();
      let detail = text;
      try {
        const json = JSON.parse(text);
        detail = json.detail || json.error?.message || text;
      } catch {}
      throw new Error(detail || `HTTP ${res.status}`);
    }
    return (await res.json()) as T;
  }

  get<T>(path: string) {
    return this.request<T>(path);
  }

  post<T>(path: string, body?: unknown) {
    const isMultipart = body instanceof FormData;
    const payload =
      body === undefined ? undefined : isMultipart ? body : JSON.stringify(body);
    return this.request<T>(path, {
      method: "POST",
      body: payload as BodyInit,
    });
  }

  postFormData<T>(path: string, formData: FormData) {
    const token = this.opts.getToken();
    return this.request<T>(path, {
      method: "POST",
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
      body: formData,
    });
  }

  put<T>(path: string, body?: unknown) {
    return this.request<T>(path, {
      method: "PUT",
      body: JSON.stringify(body),
    });
  }

  delete<T>(path: string) {
    return this.request<T>(path, { method: "DELETE" });
  }
}
