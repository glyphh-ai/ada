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
  Answer, Args, Check, Contract, Device, Edges, Facts, GqlResult, Learned, History, Loaded, LoadRecord, Model, ModelChange, NewModel,
  Prediction, Procedure, Recorded, Situation, Store, StoredRecord, Trend, Weights, When,
} from "./types.js";

export * from "./types.js";

export const DEFAULT_URL = "https://api.glyphh.ai";
/** Records one `load` call sends at once; a longer list goes in turns of this many. */
export const LOAD_BATCH = 500;

/** What a refusal's `code` can be. Others may be added. */
export type ErrorCode =
  | "E_VALIDATION" | "E_NOT_FOUND" | "E_FORBIDDEN" | "E_FAILED_PRECONDITION" | "E_UNAUTHENTICATED" | "E_PAYMENT"
  | "E_RATE_LIMITED" | "E_UNAVAILABLE" | "E_UNREACHABLE" | "E_ADA" | (string & {});

/**
 * What a refusal is, by its HTTP status, when the answer names no code of Ada's own: the gateway in front
 * of Ada (a key it does not know, a rate limit) answers in its own words.
 */
const BY_STATUS: Record<number, ErrorCode> = {
  400: "E_VALIDATION", 401: "E_UNAUTHENTICATED", 402: "E_PAYMENT", 403: "E_FORBIDDEN", 404: "E_NOT_FOUND",
  409: "E_FAILED_PRECONDITION", 429: "E_RATE_LIMITED",
};

/**
 * Ada refused, or could not be reached. `code` says which kind; `status` is the HTTP status, 0 when there
 * was none; `retryAfter` is how many seconds to wait before asking again, when Ada said (a rate limit).
 */
export class AdaError extends Error {
  readonly code: ErrorCode;
  readonly status: number;
  readonly retryAfter: number | null;
  /** When a load was refused part way: how many records had landed before the line it names. */
  readonly loaded: number | null;

  constructor(code: ErrorCode, message: string, status: number, retryAfter: number | null = null, loaded: number | null = null) {
    super(message);
    this.name = "AdaError";
    this.code = code;
    this.status = status;
    this.retryAfter = retryAfter;
    this.loaded = loaded;
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
  op<T>(op: string, args: object = {}): Promise<T> {
    return this.#send("/ada", "application/json", JSON.stringify({ op, ...without(args) }));
  }

  /**
   * A file of records into one model, streamed to POST /ada/load as NDJSON: the first line names the model, every
   * other line is one record. The server writes it in batches as it arrives, so a refusal part way names the line
   * and says how many records landed before it (`loaded` on the error). An async iterable is sent as it is read;
   * anything else is sent whole.
   */
  async upload(modelId: string, records: Iterable<LoadRecord> | AsyncIterable<LoadRecord>): Promise<Loaded> {
    const lines = (async function* () {
      yield `${JSON.stringify({ model_id: modelId })}\n`;
      for await (const r of records) yield `${JSON.stringify(r)}\n`;
    })();
    if (!(Symbol.asyncIterator in Object(records))) {
      let whole = "";
      for await (const line of lines) whole += line;
      return this.#send("/ada/load", "application/x-ndjson", whole);
    }
    const encoder = new TextEncoder();
    const body = new ReadableStream<Uint8Array>({
      async pull(controller) {
        const next = await lines.next();
        if (next.done) controller.close();
        else controller.enqueue(encoder.encode(next.value));
      },
    });
    return this.#send("/ada/load", "application/x-ndjson", body, { duplex: "half" });
  }

  async #send<T>(path: string, contentType: string, body: BodyInit, extra: Record<string, unknown> = {}): Promise<T> {
    let res: Response;
    try {
      res = await this.#fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: { "content-type": contentType, "x-glyphh-api-key": `Bearer ${this.#apiKey}`, ...this.#headers },
        body,
        ...extra,
      } as RequestInit);
    } catch (e) {
      throw new AdaError("E_UNREACHABLE", `Ada is unreachable: ${(e as Error).message}`, 0);
    }
    const answer = (await res.json().catch(() => null)) as { data?: T; error?: unknown; detail?: unknown } | null;
    if (res.ok && answer && typeof answer === "object" && "data" in answer) return answer.data as T;
    const refused = answer?.error ?? answer?.detail;
    const said = refused && typeof refused === "object" ? (refused as { code?: unknown; message?: unknown; error?: unknown; loaded?: unknown }) : {};
    const message = typeof refused === "string" ? refused : String(said.message ?? said.error ?? "") || `Ada answered ${res.status}`;
    const code = typeof said.code === "string" && said.code.startsWith("E_") ? said.code : BY_STATUS[res.status] ?? "E_ADA";
    const wait = Number(res.headers.get("retry-after"));
    throw new AdaError(code, message, res.status, res.headers.has("retry-after") && Number.isFinite(wait) ? wait : null,
                       typeof said.loaded === "number" ? said.loaded : null);
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

  /**
   * Many graded records in one call: the seed. Each is checked like `record` before any is written, so one bad
   * record refuses the call by its index and nothing lands. Up to LOAD_BATCH go in one call; a longer list goes in
   * turns, and the answer sums what landed. Call `calibrate` once at the end.
   */
  async load(records: LoadRecord[]): Promise<Loaded> {
    if (records.length === 0) throw new AdaError("E_VALIDATION", "records must be a non-empty array", 0);
    let loaded = 0;
    let last!: Loaded;
    for (let at = 0; at < records.length; at += LOAD_BATCH) {
      last = await this.#op<Loaded>("load", { records: records.slice(at, at + LOAD_BATCH) });
      loaded += last.loaded;
    }
    return { ...last, loaded };
  }

  /** A file of records, streamed as it is read: `Ada.upload` for this model. */
  stream(records: Iterable<LoadRecord> | AsyncIterable<LoadRecord>): Promise<Loaded> {
    return this.ada.upload(this.id, records);
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

  /**
   * Learn what decides from the model's wins (org admins). A flat model learns each top-level key's weight,
   * saved as its weights. A typed model learns each role's similarity weight, by path, saved in its spec.
   */
  learn(): Promise<Learned> {
    return this.#op("learn");
  }

  /** The JSON Schema of every answer and, for a typed model, of its data. */
  contract(): Promise<Contract> {
    return this.#op("contract");
  }
}
