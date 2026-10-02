// Compiled, never run: the types say what a program can rely on.
import { Ada, type Answer, type GqlResult } from "../src/index.js";

export async function route(ada: Ada, modelId: string): Promise<string> {
  const answer: Answer = await ada.model(modelId).query({ ticket: { issue: { component: "kubelet" } } });
  if (answer.act) return answer.top ?? "";
  switch (answer.reason) {
    case "unseen":
      return `never shown: ${answer.unseen.join(", ")}`;
    case "vetoed":
      return `failed before: ${answer.top}`;
    case "insufficient":
    case "uncalibrated":
    case "low_confidence":
    case "mismatch":
      return "ask someone";
    case "reflex":
      return "unreachable";
  }
}

export function summary(r: GqlResult): number {
  switch (r.statement) {
    case "find": return r.matches.length;
    case "list": return r.records.length;
    case "count": return r.count;
    case "compare": return r.score;
    case "drift": return r.drift;
    case "introspect": return r.parts.length;
    case "trend": return r.versions;
    case "predict": return r.at;
    case "aggregate": return r.groups.length;
    case "follow": return r.nodes.length;
    case "path": return r.hops ?? -1;
  }
}
