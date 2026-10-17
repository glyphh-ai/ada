/**
 * @glyphh-ai/ada — your runtime, for one key. Every call is one request to the runtime's API at its own address
 * (https://<machine>.glyphh.app), with a runtime key as the bearer. Nothing of Glyphh's is in between.
 *
 *     const incidents = new Ada().model("incidents");   // GLYPHH_RUNTIME_URL and GLYPHH_RUNTIME_KEY
 *     const r = await incidents.decide({ error: { ... } });
 *     if (r.status === "done") apply(r.data.fix);
 *     else if (r.status === "needs_input") await incidents.resume(r.run_id, await ask(r.prompt));
 */

export type Status = "done" | "needs_input" | "needs_approval" | "failed" | "expired" | "cancelled";

/** What a run answered: its status, and its data (the function's answer when done; why, when failed). */
export interface Result<T = unknown> {
  run_id: string;
  model: string;
  version: number;
  function: string;
  status: Status;
  data: T;
  step: string | null;
  prompt: string | null;
  seconds: number;
}

/** What an Ada model's `query` answers. Trust `top` when `sufficient` and `confidence` clears your bar. */
export interface Answer {
  sufficient: boolean;
  confidence: number;
  support: number;
  count: number;
  top: string | null;
  outcomes: Record<string, number>;
  receipts: { id: string; sim: number; outcome: string | null; hash: string; author: string | null }[];
}

export interface ModelInfo {
  name: string;
  [key: string]: unknown;
}

/** Sends one request; give your own to use another HTTP client. */
export type Transport = (method: string, url: string, headers: Record<string, string>, body?: string) => Promise<{ status: number; body: string }>;

export interface AdaOptions {
  /** The runtime's address; else GLYPHH_RUNTIME_URL. */
  url?: string;
  /** A runtime key; else GLYPHH_RUNTIME_KEY. */
  key?: string;
  transport?: Transport;
}

export class AdaError extends Error {
  constructor(
    readonly code: string,
    message: string,
    readonly status = 0,
    readonly detail: Record<string, unknown> = {},
  ) {
    super(message);
    this.name = "AdaError";
  }
}

const BY_STATUS: Record<number, string> = { 401: "E_AUTH", 402: "E_LIMIT", 404: "E_NOT_FOUND" };

function env(name: string): string | undefined {
  const p = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return p?.env?.[name];
}

const fetchTransport: Transport = async (method, url, headers, body) => {
  const res = await fetch(url, body === undefined ? { method, headers } : { method, headers, body });
  return { status: res.status, body: await res.text() };
};

export class Ada {
  readonly url: string;
  readonly #key: string;
  readonly #send: Transport;

  constructor(options: AdaOptions = {}) {
    this.url = (options.url ?? env("GLYPHH_RUNTIME_URL") ?? "").replace(/\/+$/, "");
    this.#key = options.key ?? env("GLYPHH_RUNTIME_KEY") ?? "";
    if (!this.url || !this.#key) {
      throw new AdaError("E_CONFIG", "the runtime's address and a runtime key are needed: GLYPHH_RUNTIME_URL, GLYPHH_RUNTIME_KEY");
    }
    this.#send = options.transport ?? fetchTransport;
  }

  async call<T = unknown>(method: string, path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = { authorization: `Bearer ${this.#key}`, accept: "application/json" };
    let raw: string | undefined;
    if (body !== undefined) {
      headers["content-type"] = "application/json";
      raw = JSON.stringify(body);
    }
    let status: number;
    let text: string;
    try {
      ({ status, body: text } = await this.#send(method, `${this.url}${path}`, headers, raw));
    } catch (e) {
      throw new AdaError("E_UNREACHABLE", `the runtime could not be reached: ${(e as Error).message}`);
    }
    let parsed: unknown = {};
    try {
      parsed = text ? JSON.parse(text) : {};
    } catch {
      parsed = {};
    }
    if (status >= 400) {
      const err = ((parsed as { error?: Record<string, unknown> }).error ?? {}) as Record<string, unknown>;
      const { code, message, ...detail } = err;
      throw new AdaError(
        (code as string | undefined) ?? BY_STATUS[status] ?? "E_RUNTIME",
        (message as string | undefined) ?? `the runtime answered ${status}`,
        status,
        detail,
      );
    }
    return parsed as T;
  }

  /** The models deployed: each one's versions, the active one and its functions. */
  async models(): Promise<ModelInfo[]> {
    return (await this.call<{ models: ModelInfo[] }>("GET", "/v1/models")).models;
  }

  /** A commit of a model's GitHub repo (owner/name) as its next version; a token for a private repo. */
  deploy(repo: string, commit: string, token?: string): Promise<{ model: string; version: number; commit_sha: string }> {
    return this.call("POST", "/v1/models/deploy", token === undefined ? { repo, commit } : { repo, commit, token });
  }

  model(name: string): AdaModel {
    return new AdaModel(this, name);
  }

  /** A run as the runtime shows it: status, function, times, the last Result. */
  run(runId: string): Promise<Record<string, unknown>> {
    return this.call("GET", `/v1/runs/${encodeURIComponent(runId)}`);
  }

  async log(runId: string): Promise<unknown[]> {
    return (await this.call<{ lines: unknown[] }>("GET", `/v1/runs/${encodeURIComponent(runId)}/log`)).lines;
  }

  /** Answer a run that waits (`needs_input`, `needs_approval`): the next Result. */
  resume<T = unknown>(runId: string, answer: unknown): Promise<Result<T>> {
    return this.call("POST", `/v1/runs/${encodeURIComponent(runId)}/continue`, { answer });
  }

  cancel<T = unknown>(runId: string): Promise<Result<T>> {
    return this.call("POST", `/v1/runs/${encodeURIComponent(runId)}/cancel`, {});
  }
}

export interface CallOptions {
  version?: number;
  /** Who asks, 0 to 1: the model's secured layers are cut from the answer below their weight. */
  clearance?: number;
}

/** One model on the runtime. `call` reaches any function; the named ones are an Ada model's own. */
export class AdaModel {
  constructor(
    readonly ada: Ada,
    readonly name: string,
  ) {}

  call<T = unknown>(fn = "start", inputs: Record<string, unknown> = {}, options: CallOptions = {}): Promise<Result<T>> {
    const body: Record<string, unknown> = { model: this.name, function: fn, inputs };
    if (options.version !== undefined) body.version = options.version;
    if (options.clearance !== undefined) body.clearance = options.clearance;
    return this.ada.call("POST", "/v1/runs", body);
  }

  resume<T = unknown>(runId: string, answer: unknown): Promise<Result<T>> {
    return this.ada.resume(runId, answer);
  }

  record(data: Record<string, unknown>, outcome?: string): Promise<Result<{ id: string }>> {
    return this.call("record", outcome === undefined ? { data } : { data, outcome });
  }

  query(data: Record<string, unknown>, options: CallOptions = {}): Promise<Result<Answer>> {
    return this.call("query", { data }, options);
  }

  facts(data: Record<string, unknown>, options: CallOptions & { top?: number } = {}): Promise<Result<unknown[]>> {
    const { top, ...rest } = options;
    return this.call("facts", top === undefined ? { data } : { data, top }, rest);
  }

  probabilities(data: Record<string, unknown>, options: CallOptions = {}): Promise<Result<Record<string, number>>> {
    return this.call("probabilities", { data }, options);
  }

  decide<T = unknown>(data: Record<string, unknown>, options: CallOptions = {}): Promise<Result<T>> {
    return this.call("decide", { data }, options);
  }

  /** The model weighs its roles from its own precedents and calibrates; `sample` is how many of the newest to calibrate on. */
  learn(options: CallOptions & { sample?: number } = {}): Promise<Result<{ roles: Record<string, number>; tau: number; precedents: number }>> {
    const { sample, ...rest } = options;
    return this.call("learn", sample === undefined ? {} : { sample }, rest);
  }
}
