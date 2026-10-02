/**
 * Ada for TypeScript and JavaScript.
 *
 * One endpoint, POST /ada, on a Glyphh API key. An agent builds a model: its
 * types, its records, its rules. This client is what the software that then
 * runs it calls. Every answer is typed, the same input always gives the same
 * answer, and no language model is in the path.
 *
 *     const ada = new Ada({ apiKey: process.env.GLYPHH_API_KEY });
 *     const answer = await ada.model("am_0123456789ab").query({ ticket: { issue: { component: "kubelet" } } });
 *     if (answer.act) route(answer.top);
 */

import type {
  Answer, Args, Check, Contract, Device, Edges, Facts, GqlResult, History, Model, ModelChange, NewModel, Prediction, Procedure,
  Recorded, Situation, Store, StoredRecord, Trend, Weights, When,
} from "./types.js";

export * from "./types.js";

export const DEFAULT_URL = "https://api.glyphh.ai";

/** What a refusal's `code` can be. Others may be added. */
export type ErrorCode =
  | "E_VALIDATION" | "E_NOT_FOUND" | "E_FORBIDDEN" | "E_FAILED_PRECONDITION" | "E_UNAUTHENTICATED" | "E_PAYMENT"
  | "E_UNAVAILABLE" | "E_UNREACHABLE" | "E_ADA" | (string & {});

/** Ada refused, or could not be reached. `code` says which kind; `status` is the HTTP status, 0 when there was none. */
export class AdaError extends Error {
  readonly code: ErrorCode;
  readonly status: number;

  constructor(code: ErrorCode, message: string, status: number) {
    super(message);
    this.name = "AdaError";
    this.code = code;
    this.status = status;
  }
}

export interface AdaOptions {
  /** A Glyphh API key. Defaults to the GLYPHH_API_KEY environment variable. */
  apiKey?: string;
  /** Defaults to GLYPHH_URL, else https://api.glyphh.ai. */
  baseUrl?: string;
  /** The fetch to use. Defaults to the global one. */
  fetch?: typeof fetch;
  /** Extra headers on every call. */
  headers?: Record<string, string>;
}

function env(name: string): string | undefined {
  const held = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env;
  return held?.[name] || undefined;
}

function without<T extends object>(args: T): Partial<T> {
  return Object.fromEntries(Object.entries(args).filter(([, v]) => v !== undefined)) as Partial<T>;
}

export class Ada {
  readonly baseUrl: string;
  readonly #apiKey: string;
  readonly #fetch: typeof fetch;
  readonly #headers: Record<string, string>;

  constructor(options: AdaOptions = {}) {
    const key = options.apiKey ?? env("GLYPHH_API_KEY");
    if (!key) throw new AdaError("E_UNAUTHENTICATED", "an API key is required: pass apiKey, or set GLYPHH_API_KEY", 0);
    this.#apiKey = key;
    this.baseUrl = (options.baseUrl ?? env("GLYPHH_URL") ?? DEFAULT_URL).replace(/\/+$/, "");
    this.#fetch = options.fetch ?? ((...a) => fetch(...a));
    this.#headers = options.headers ?? {};
  }

  /** One Ada operation by name: the tool's name without `ada_`, and its arguments. The typed methods below call this. */
  async op<T>(op: string, args: object = {}): Promise<T> {
    let res: Response;
    try {
      res = await this.#fetch(`${this.baseUrl}/ada`, {
        method: "POST",
        headers: { "content-type": "application/json", "x-glyphh-api-key": `Bearer ${this.#apiKey}`, ...this.#headers },
        body: JSON.stringify({ op, ...without(args) }),
      });
    } catch (e) {
      throw new AdaError("E_UNREACHABLE", `Ada is unreachable: ${(e as Error).message}`, 0);
    }
    const body = (await res.json().catch(() => null)) as { data?: T; error?: unknown; detail?: unknown } | null;
    if (res.ok && body && typeof body === "object" && "data" in body) return body.data as T;
    const refused = body?.error ?? body?.detail;
    const said = refused && typeof refused === "object" ? (refused as { code?: unknown; message?: unknown; error?: unknown }) : {};
    const message = typeof refused === "string" ? refused : String(said.message ?? said.error ?? "") || `Ada answered ${res.status}`;
    const code = typeof said.code === "string" ? said.code : res.status === 401 ? "E_UNAUTHENTICATED" : res.status === 402 ? "E_PAYMENT" : "E_ADA";
    throw new AdaError(code, message, res.status);
  }

  /** One model, by its id (am_ and 12 hex digits). */
  model(modelId: string): AdaModel {
    return new AdaModel(this, modelId);
  }

  /** The organization's models. */
  async models(): Promise<Model[]> {
    return (await this.op<{ models: Model[] }>("models")).models;
  }

  /** Create a model (org admins). With a `spec` it is typed; without, it takes any JSON. */
  async createModel(model: NewModel): Promise<AdaModel & { created: Model }> {
    const created = await this.op<Model>("create_model", model);
    return Object.assign(this.model(created.model_id), { created });
  }

  /** The registered devices that can hold a local model's records. */
  async devices(): Promise<Device[]> {
    return (await this.op<{ devices: Device[] }>("devices")).devices;
  }

  /** The JSON Schema of every answer, and what each reason means. */
  contract(): Promise<Contract> {
    return this.op("contract");
  }
}

export class AdaModel {
  constructor(private readonly ada: Ada, readonly id: string) {}

  #op<T>(op: string, args: object = {}): Promise<T> {
    return this.ada.op<T>(op, { model_id: this.id, ...args });
  }

  // ── ask ──

  /** What situations like this one resolve to, and whether to act on it. Branch on `act`; read `reason` when it is false. */
  query(situation: Situation, options: { weights?: Weights } = {}): Promise<Answer> {
    return this.#op("query", { situation, ...options });
  }

  /** Why: the nearest situations among the wins and the failures, each as a fact tree. */
  facts(situation: Situation, options: { weights?: Weights; top?: number } = {}): Promise<Facts> {
    return this.#op("facts", { situation, ...options });
  }

  /** How close the nearest win and the nearest failure of one outcome are. */
  check(situation: Situation, outcome: string, options: { weights?: Weights } = {}): Promise<Check> {
    return this.#op("check", { situation, outcome, ...options });
  }

  // ── record ──

  /** A graded win: this outcome worked here. `at` is when it happened; now when left out. */
  record(situation: Situation, outcome: string, options: { at?: When } = {}): Promise<Recorded> {
    return this.#op("record", { situation, outcome, ...options });
  }

  /** A graded failure: this outcome failed, or was rejected, here. */
  veto(situation: Situation, outcome: string, options: { at?: When } = {}): Promise<Recorded> {
    return this.#op("veto", { situation, outcome, ...options });
  }

  // ── one thing, in time ──

  /** Every record of exactly this situation, oldest first. In a model with key parts: every version of the thing, each with what changed. */
  history(situation: Situation): Promise<History> {
    return this.#op("history", { situation });
  }

  /** Where one thing has been going. `thing` carries its key parts. */
  trend(thing: Situation): Promise<Trend> {
    return this.#op("trend", { situation: thing });
  }

  /** One thing's next version, and what the model's other things say of it. */
  predict(thing: Situation, options: { at?: When } = {}): Promise<Prediction> {
    return this.#op("predict", { situation: thing, ...options });
  }

  /** One thing's edges: the things nearest it at a level, the change between its versions, and its relations. */
  edges(thing: Situation, options: { level?: string; top?: number } = {}): Promise<Edges> {
    return this.#op("edges", { situation: thing, ...options });
  }

  // ── GQL ──

  /** One GQL statement, with its $parameters when it takes any. Narrow the result on `statement`. */
  gql<R extends GqlResult = GqlResult>(query: string, args?: Args): Promise<R> {
    return this.#op("gql", { query, args });
  }

  /** Run a stored procedure by name. */
  call<R extends GqlResult = GqlResult>(name: string, args?: Args): Promise<R> {
    return this.#op("call", { name, args });
  }

  /** The model's stored procedures. */
  async procedures(): Promise<Procedure[]> {
    return (await this.#op<{ procedures: Procedure[] }>("procedures")).procedures;
  }

  /** Save a stored procedure, new or replaced (org admins). The statement is checked now. */
  saveProcedure(name: string, query: string, description?: string): Promise<Procedure> {
    return this.#op("save_procedure", { name, query, description });
  }

  /** Delete a stored procedure (org admins). */
  deleteProcedure(name: string): Promise<{ model_id: string; name: string; deleted: true }> {
    return this.#op("delete_procedure", { name });
  }

  // ── the model and its records ──

  /** The model with its record counts and cache counts. */
  async info(): Promise<Model> {
    const found = (await this.#op<{ models: Model[] }>("models")).models[0];
    if (!found) throw new AdaError("E_NOT_FOUND", `no Ada model ${this.id}`, 404);
    return found;
  }

  /** Rename the model, or set its weights, scale, edges or (while it has no records) its spec (org admins). */
  update(change: ModelChange): Promise<Model> {
    return this.#op("update_model", change);
  }

  /** Delete the model and its records, wherever they live (org admins). */
  delete(): Promise<{ model_id: string; deleted: true; records: number }> {
    return this.#op("delete_model");
  }

  /** The model's records, newest first. */
  records(options: { store?: Store; offset?: number; limit?: number } = {}): Promise<{ model_id: string; total: number; records: StoredRecord[] }> {
    return this.#op("records", options);
  }

  /** Correct a record (org admins). */
  editRecord(recordId: string, change: { situation?: Situation; outcome?: string }): Promise<{ model_id: string; id: string; updated: true }> {
    return this.#op("edit_record", { record_id: recordId, ...change });
  }

  /** Delete one record (org admins). */
  deleteRecord(recordId: string): Promise<{ model_id: string; id: string; deleted: true }> {
    return this.#op("delete_record", { record_id: recordId });
  }

  /** Fit the model's probabilities to its own records (10 or more wins). */
  calibrate(): Promise<{ model_id: string; tau: number; size: number }> {
    return this.#op("calibrate");
  }

  /** Learn which keys decide from a flat model's wins, and save them as its weights (org admins). */
  learn(): Promise<{ model_id: string; weights: Weights; tau: number; size: number; held_out_accuracy: { before: number; after: number } }> {
    return this.#op("learn");
  }

  /** The JSON Schema of every answer and, for a typed model, of its data. */
  contract(): Promise<Contract> {
    return this.#op("contract");
  }
}
