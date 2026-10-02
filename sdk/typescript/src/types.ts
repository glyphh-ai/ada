/** What Ada takes and returns. These follow contract version 2 (contract/contract.json in this repository). */

export type Scalar = string | number | boolean;
/** Data sent to a model. A flat model takes any JSON object; a typed one takes its spec's shape: layer > segment > role. */
export type Situation = Record<string, unknown>;
/** A time: epoch seconds, or an ISO 8601 string. */
export type When = number | string;

/** How much each key decides. A number weighs a key; an object weighs inside it, its own weight under "*". `"~all": true` makes an object's keys decide together. */
export interface Weights {
  [key: string]: number | boolean | Weights;
}

/** What `act` takes for a model. */
export interface Scale {
  similarity: number;
  confidence: number;
  records: number;
  exact: boolean;
  must_match: string[];
  /** True lets the model act on a situation that carries a key or value none of its wins recorded. */
  unseen: boolean;
}

// ── a typed model's spec ─────────────────────────────────────────────────────

export type RoleType = "category" | "set" | "number" | "text" | "boolean" | "time" | "ref";

export interface Role {
  name: string;
  type?: RoleType;
  /** What a category or a set may hold. */
  values?: Scalar[];
  /** A number's scale. */
  numeric_config?: { min_value: number; max_value: number; bin_width: number };
  text_encoding?: "bag_of_words" | "ngrams";
  /** True for the roles that say which thing a record is. Records that share them are versions of one thing. */
  key_part?: boolean;
  similarity_weight?: number;
  security_weight?: number;
  /** Other names the role answers to in data. */
  lexicons?: string[];
  /** A ref role: the model it points into. Its own model when left out. */
  to?: string;
  /** A ref role: takes a list of things. */
  many?: boolean;
  /** A ref role: a record that names a thing not on record is refused. */
  strict?: boolean;
}

export interface Segment {
  name: string;
  roles: Role[];
  similarity_weight?: number;
  security_weight?: number;
  /** The roles decide together: all the question gives have to agree. */
  together?: boolean;
}

export interface Layer {
  name: string;
  segments: Segment[];
  similarity_weight?: number;
  security_weight?: number;
}

export interface Spec {
  name: string;
  layers: Layer[];
  /** layer.segment.role of the role that carries when a record happened. */
  temporal_source?: string;
  dimension?: number;
  seed?: number;
  similarity_weight?: number;
  security_weight?: number;
  output?: { report: string[] };
}

// ── models and records ───────────────────────────────────────────────────────

export interface Model {
  model_id: string;
  name: string;
  storage: "cloud" | "local";
  device_id: string | null;
  weights: Weights | null;
  tau: number | null;
  calibrated: boolean;
  scale: Scale;
  /** The spec of a typed model. Null for a flat one. */
  spec: Spec | null;
  edges: { neural: [number, number]; semantic: [number, number] };
  data_version: number;
  created_at: string;
  updated_at: string;
  /** Present when one model is asked for. */
  wins?: number;
  vetoes?: number;
  things?: number;
  cache?: CacheStats;
  /** A cloud model: the records its working copy has taken in, and how many the copy kept in the database covers. */
  index?: { records: number; kept: number };
}

/** What `learn` found. `weights` is a flat model's weights tree, or a typed model's role weights by path. */
export interface Learned {
  model_id: string;
  weights: Weights | Record<string, number>;
  tau: number;
  size: number;
  held_out_accuracy: { before: number; after: number };
  /** A typed model: the wins held out, and the wins they were scored against. */
  held?: number;
  scored?: number;
}

export interface CacheStats {
  hits: number;
  misses: number;
  invalidations: number;
  evictions: number;
  hit_rate: number;
  entries: number;
}

export interface NewModel {
  name: string;
  storage: "cloud" | "local";
  device_id?: string;
  weights?: Weights;
  spec?: Spec;
}

export interface ModelChange {
  name?: string;
  weights?: Weights | null;
  scale?: Partial<Scale>;
  spec?: Spec;
  edges?: { neural?: [number, number]; semantic?: [number, number] };
}

export type Store = "wins" | "vetoes";

export interface StoredRecord {
  seq: number;
  id: string;
  key: string;
  store: Store;
  situation: Situation;
  outcome: string;
  author: string;
  ts: number;
}

export interface Recorded {
  model_id: string;
  store: Store;
  id: string;
  key: string;
  ts: number;
  /** In a model with key parts: which version of the thing this record is, in time. */
  version: number | null;
  size: number;
}

export interface Device {
  device_id: string;
  owner_user_id: string;
  name: string;
  created_at: string;
  online: boolean;
}

// ── a query's answer ─────────────────────────────────────────────────────────

export type Reason = "reflex" | "insufficient" | "unseen" | "vetoed" | "uncalibrated" | "mismatch" | "low_confidence";

/** One situation on record with one outcome, however many times it was recorded. */
export interface Receipt {
  key: string;
  outcome: string;
  score: number;
  /** The same number as `score`, under its earlier name. */
  sim: number;
  count: number;
  first: number;
  latest: number;
  id: string;
  hash: string;
  author: string;
}

export interface Answer {
  contract: number;
  model_id: string;
  key: string;
  /** Branch on this. True only when `reason` is "reflex". */
  act: boolean;
  reason: Reason;
  /** The leading outcome. Null when nothing is close enough. */
  top: string | null;
  sufficient: boolean;
  exact: boolean;
  /** What the situation carries that no win recorded: a key ("force") or a value ("target=main"). */
  unseen: string[];
  confidence: number;
  support: number;
  count: number;
  outcomes: Record<string, number>;
  receipts: Receipt[];
  veto_match: number;
  lifted: boolean;
  calibrated: boolean;
  scale: Scale;
}

export type Match = "same" | "lexical" | "partial" | "near" | "different";

export interface Fact {
  key: string;
  weight: number;
  score: number;
  share: number;
  sim: number;
  status: "both" | "query_only" | "record_only";
  match?: Match;
  query?: unknown;
  record?: unknown;
  children?: Fact[];
}

export interface FactTree {
  id: string;
  key: string;
  outcome: string | null;
  probability: number;
  score: number;
  sim: number;
  count: number;
  first: number;
  latest: number;
  author: string;
  ts: number;
  hash: string;
  tree: Fact;
}

export interface Facts {
  model_id: string;
  wins: FactTree[];
  vetoes: FactTree[];
}

export interface Check {
  model_id: string;
  outcome: string;
  win_match: number;
  veto_match: number;
}

// ── time ─────────────────────────────────────────────────────────────────────

/** One role that differs between two versions of a thing. */
export interface Change {
  path: string;
  from: unknown;
  to: unknown;
  score: number;
}

export interface Observation {
  id: string;
  store: Store;
  outcome: string;
  author: string;
  ts: number;
  /** In a model with key parts. */
  version?: number;
  data?: Situation;
  changes?: Change[];
  change?: number;
}

export interface History {
  model_id: string;
  key: string;
  count: number;
  first: number | null;
  latest: number | null;
  observations: Observation[];
}

export interface RoleTrend {
  type: string;
  first: unknown;
  latest: unknown;
  changes: number;
  since: number;
  slope_per_day?: number | null;
  fit?: number | null;
  direction?: "up" | "down" | "flat";
}

export interface Trend {
  model_id: string;
  key: string;
  versions: number;
  first: number;
  latest: number;
  drift: number;
  pace: number;
  period: number | null;
  roles: Record<string, RoleTrend>;
}

export interface RolePrediction {
  from: unknown;
  to: unknown;
  basis: "trend" | "own_history" | "model" | "unchanged";
  confidence: number | null;
}

export interface Prediction {
  model_id: string;
  key: string;
  versions: number;
  at: number;
  data: Situation;
  roles: Record<string, RolePrediction>;
  /** What a query returns for the predicted version, the thing itself left out. */
  answer: Answer;
}

// ── edges and the graph ──────────────────────────────────────────────────────

/** One relation: the thing that has the ref role, the role's path, and the thing it names. */
export interface Relation {
  from: { model_id: string; key: string };
  relation: string;
  to: { model_id: string; key: string | null; parts: Scalar[] };
  /** True when the relation names a thing that is not on record. */
  missing: boolean;
}

/** A thing: its newest version, as the reader may see it. */
export interface GraphNode {
  model_id: string;
  key: string;
  missing: boolean;
  id?: string;
  store?: Store;
  outcome?: string;
  ts?: number;
  data?: Situation;
  hops?: number;
}

export interface Edges {
  model_id: string;
  key: string;
  level: string;
  neural: { target: string; id: string; store: Store; outcome: string | null; score: number }[];
  temporal: { from: number; to: number; ts: number; change: number; changes: Change[] }[];
  relations: { out: Relation[]; in: Relation[] };
}

// ── GQL ──────────────────────────────────────────────────────────────────────

interface Ran<S extends string> {
  statement: S;
  model_id: string;
  /** True when this is the answer given before, and nothing it stood on has changed. */
  cached: boolean;
  /** The stored procedure that was called, when one was. */
  procedure?: string;
}

export interface FindMatch {
  key: string;
  id: string;
  store: Store;
  outcome: string | null;
  score: number;
  count: number;
  first: number;
  latest: number;
}

export interface FindResult extends Ran<"find"> {
  level: string;
  threshold: number;
  count: number;
  matches: FindMatch[];
}

export interface ListResult extends Ran<"list"> {
  count: number;
  records: { key: string; id: string; store: Store; outcome: string | null; author: string; ts: number; data: Situation }[];
}

export interface CountResult extends Ran<"count"> {
  count: number;
}

export interface CompareResult extends Ran<"compare"> {
  level: string;
  score: number;
  tree: Fact;
}

export interface DriftResult extends Ran<"drift"> {
  level: string;
  drift: number;
  threshold: number | null;
  drifted: boolean | null;
  changes: Change[];
}

export interface IntrospectResult extends Ran<"introspect"> {
  by: "layer" | "segment" | "role";
  parts: { path: string; weight: number; value: unknown }[];
}

export interface TrendResult extends Ran<"trend">, Omit<Trend, "model_id"> {
  level: string;
  over: number;
  alert: { on: string; value: number; fired: boolean } | null;
}

export interface PredictResult extends Ran<"predict">, Omit<Prediction, "model_id"> {}

export interface AggregateResult extends Ran<"aggregate"> {
  function: "count" | "avg" | "min" | "max" | "sum";
  path: string | null;
  groups: { by: Record<string, unknown>; value: number; count: number }[];
}

export interface FollowResult extends Ran<"follow"> {
  from: GraphNode;
  direction: "out" | "in" | "both";
  depth: number;
  count: number;
  nodes: (GraphNode & { hops: number })[];
  edges: Relation[];
  truncated: boolean;
}

export interface PathResult extends Ran<"path"> {
  found: boolean;
  hops: number | null;
  direction: "out" | "in" | "both";
  depth: number;
  nodes: GraphNode[];
  edges: Relation[];
}

/** What one GQL statement returns. `statement` says which. */
export type GqlResult =
  | FindResult | ListResult | CountResult | CompareResult | DriftResult | IntrospectResult
  | TrendResult | PredictResult | AggregateResult | FollowResult | PathResult;

/** A statement's $parameters, by name. */
export type Args = Record<string, Scalar | Situation>;

export interface Procedure {
  name: string;
  query: string;
  description: string;
  params: string[];
  updated_at: string;
}

/** The contract the server publishes: the JSON Schema of every answer, and of a typed model's data. */
export interface Contract {
  contract: number;
  reasons: Record<Reason, string>;
  query: object;
  facts: object;
  history: object;
  trend: object;
  predict: object;
  gql: object;
  edges: object;
  /** With a typed model's id: the shape of its data. */
  data?: object;
}
