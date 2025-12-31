from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class SimilarityMetrics:
    raw_cosine: float
    shifted_cosine01: float


def _bipolar_raw_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError("Vector shapes must match")
    n = int(a.size)
    if n == 0:
        return 0.0
    a_i = a.astype(np.int32, copy=False)
    b_i = b.astype(np.int32, copy=False)
    return float((a_i * b_i).sum() / n)


def compute_pair_metrics(a: np.ndarray, b: np.ndarray) -> SimilarityMetrics:
    raw = _bipolar_raw_cosine(a, b)
    return SimilarityMetrics(raw_cosine=raw, shifted_cosine01=(raw + 1.0) / 2.0)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def _extract_taxonomy(semantic: Dict[str, Any]) -> List[str]:
    taxonomy = semantic.get("taxonomy")
    if isinstance(taxonomy, list):
        return [str(x) for x in taxonomy if x is not None]
    if isinstance(taxonomy, str) and taxonomy.strip():
        return [taxonomy.strip()]
    return []


def _matches_filter(
    name: str,
    semantic: Dict[str, Any],
    *,
    name_prefix: str | None = None,
    taxonomy_includes: str | None = None,
    semantic_equals: Dict[str, Any] | None = None,
) -> bool:
    if name_prefix and not name.startswith(name_prefix):
        return False
    if taxonomy_includes:
        want = taxonomy_includes.lower()
        tax = [t.lower() for t in _extract_taxonomy(semantic)]
        if want not in tax:
            return False
    if semantic_equals:
        for key, value in semantic_equals.items():
            if semantic.get(key) != value:
                return False
    return True


def _label_for(semantic: Dict[str, Any], name: str, label_role: str | None) -> str:
    if label_role:
        v = semantic.get(label_role)
        if v is not None and str(v).strip() != "":
            return str(v)
    return name


def _expected_neighbors(
    semantic: Dict[str, Any],
    existing_names: set[str],
    expected_neighbor_roles: Sequence[str] | None,
) -> List[str]:
    if not expected_neighbor_roles:
        return []
    out: List[str] = []
    for role in expected_neighbor_roles:
        v = semantic.get(role)
        if isinstance(v, str) and v in existing_names:
            out.append(v)
    return out


def _compute_similarity_matrix(vecs: np.ndarray) -> np.ndarray:
    if vecs.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    n = vecs.shape[1]
    dots = vecs.astype(np.int16, copy=False) @ vecs.T.astype(np.int16, copy=False)
    raw = dots.astype(np.float64) / float(n)
    shifted = (raw + 1.0) / 2.0
    return shifted


def build_similarity_report(
    glyphs: Iterable[Tuple[str, Dict[str, Any], bytes]],
    *,
    groups: Sequence[Dict[str, Any]],
    pairs: Sequence[Tuple[str, str]] | None = None,
    expected_space_id: str | None = None,
    enforce_single_space: bool = True,
) -> Dict[str, Any]:
    """
    glyphs: iterable of (name, semantic, cortex_bytes) where cortex is bipolar int8 bytes.
    """
    input_rows = [(name, semantic or {}, cortex) for name, semantic, cortex in glyphs]
    space_counts: Dict[str, int] = {}
    missing_space_id = 0
    for _, semantic, _ in input_rows:
        sid = semantic.get("_space_id")
        if isinstance(sid, str) and sid:
            space_counts[sid] = space_counts.get(sid, 0) + 1
        else:
            missing_space_id += 1

    rows = input_rows
    excluded = 0
    if enforce_single_space and expected_space_id:
        filtered: List[Tuple[str, Dict[str, Any], bytes]] = []
        for name, semantic, cortex in input_rows:
            sid = semantic.get("_space_id")
            if sid == expected_space_id:
                filtered.append((name, semantic, cortex))
            else:
                excluded += 1
        rows = filtered
    by_name = {name: (semantic, cortex) for name, semantic, cortex in rows}

    pair_results: List[Dict[str, Any]] = []
    for left, right in pairs or []:
        left_row = by_name.get(left)
        right_row = by_name.get(right)
        if not left_row or not right_row:
            pair_results.append(
                {
                    "left": left,
                    "right": right,
                    "error": "glyph not found",
                }
            )
            continue
        a = np.frombuffer(left_row[1], dtype=np.int8)
        b = np.frombuffer(right_row[1], dtype=np.int8)
        metrics = compute_pair_metrics(a, b)
        pair_results.append(
            {
                "left": left,
                "right": right,
                "raw_cosine": metrics.raw_cosine,
                "shifted_cosine01": metrics.shifted_cosine01,
            }
        )

    group_results: List[Dict[str, Any]] = []
    report_lines: List[str] = []

    if enforce_single_space and expected_space_id:
        report_lines.append(f"Expected space_id: {expected_space_id}")
        if missing_space_id:
            report_lines.append(f"WARNING: {missing_space_id} glyph(s) missing _space_id")
        if excluded:
            report_lines.append(f"WARNING: excluded {excluded} glyph(s) not in expected space")

    if pair_results:
        for entry in pair_results:
            if entry.get("error"):
                report_lines.append(f"Similarity ({entry['left']} vs {entry['right']}): ERROR {entry['error']}")
                continue
            report_lines.append(
                f"Similarity ({entry['left']} vs {entry['right']}): {entry['shifted_cosine01']:.6f}"
            )
            report_lines.append(f"Raw cosine        : {entry['raw_cosine']:.6f}")
            report_lines.append(f"Shifted cosine01  : {entry['shifted_cosine01']:.6f}")

    for group in groups:
        group_name = str(group.get("name") or "group")
        name_prefix = group.get("name_prefix")
        taxonomy_includes = group.get("taxonomy_includes")
        semantic_equals = group.get("semantic_equals")
        label_role = group.get("label_role")
        order_role = group.get("order_role")
        expected_neighbor_roles = group.get("expected_neighbor_roles") or []
        top_k = int(group.get("top_k") or 3)
        far_offset = group.get("far_offset")

        selected: List[Tuple[str, Dict[str, Any], bytes]] = []
        for name, semantic, cortex in rows:
            if _matches_filter(
                name,
                semantic,
                name_prefix=name_prefix,
                taxonomy_includes=taxonomy_includes,
                semantic_equals=semantic_equals,
            ):
                selected.append((name, semantic, cortex))

        if not selected:
            group_results.append({"name": group_name, "count": 0, "neighbors": [], "notes": "no glyphs matched"})
            continue

        names = [n for n, _, _ in selected]
        semantics = {n: sem for n, sem, _ in selected}
        existing = set(names)
        labels = {n: _label_for(semantics[n], n, label_role) for n in names}
        name_to_idx = {n: i for i, n in enumerate(names)}

        mat = np.stack([np.frombuffer(c, dtype=np.int8) for _, _, c in selected], axis=0)
        sim_full = _compute_similarity_matrix(mat)
        sim_for_neighbors = sim_full.copy()
        np.fill_diagonal(sim_for_neighbors, -np.inf)
        neighbors: Dict[str, List[Tuple[str, float]]] = {}
        top1_ok = 0
        top3_ok = 0
        total = 0
        for i, name in enumerate(names):
            scores = sim_for_neighbors[i]
            idxs = np.argsort(scores)[::-1][: max(1, top_k)]
            items: List[Tuple[str, float]] = []
            for j in idxs:
                other = names[int(j)]
                items.append((other, float(scores[int(j)])))
            neighbors[name] = items

            expected = _expected_neighbors(semantics[name], existing, expected_neighbor_roles)
            if expected:
                total += 1
                if items and items[0][0] in expected:
                    top1_ok += 1
                if any(item[0] in expected for item in items[: min(3, len(items))]):
                    top3_ok += 1

        sorted_names = names[:]
        if order_role:
            with_order: List[Tuple[float, str]] = []
            without: List[str] = []
            for name in names:
                v = _safe_float(semantics[name].get(order_role))
                if v is None:
                    without.append(name)
                else:
                    with_order.append((v, name))
            with_order.sort(key=lambda x: x[0])
            sorted_names = [n for _, n in with_order] + sorted(without)
        else:
            sorted_names = sorted(sorted_names, key=lambda n: str(labels[n]))

        adjacent_sims: List[float] = []
        for a_name, b_name in zip(sorted_names, sorted_names[1:]):
            ia = name_to_idx[a_name]
            ib = name_to_idx[b_name]
            adjacent_sims.append(float(sim_full[ia, ib]))

        if far_offset is None:
            far_offset = max(1, len(sorted_names) // 2)
        far_offset = int(far_offset)
        far_sims: List[float] = []
        for idx in range(0, len(sorted_names) - far_offset):
            a_name = sorted_names[idx]
            b_name = sorted_names[idx + far_offset]
            ia = name_to_idx[a_name]
            ib = name_to_idx[b_name]
            far_sims.append(float(sim_full[ia, ib]))

        adjacent_avg = float(np.mean(adjacent_sims)) if adjacent_sims else 0.0
        far_avg = float(np.mean(far_sims)) if far_sims else 0.0
        gap = adjacent_avg - far_avg

        report_lines.append("")
        report_lines.append(f"{group_name} nearest neighbors (top {top_k}):")
        for name in sorted(sorted_names, key=lambda n: str(labels[n])):
            items = neighbors.get(name, [])
            rendered = ", ".join(
                f"{labels[other]}({score:.3f})" for other, score in items
            )
            report_lines.append(f"  {labels[name]}: {rendered}")

        report_lines.append("")
        report_lines.append(f"Average sim({group_name}_i, {group_name}_i+1): {adjacent_avg:.6f}")
        report_lines.append(f"Average sim({group_name}_i, {group_name}_i+{far_offset}): {far_avg:.6f}")
        report_lines.append(f"Gap        : {gap:.6f}")
        if total:
            report_lines.append(
                f"Top1={top1_ok/total:.2%} | Top3={top3_ok/total:.2%} | evaluated={total}"
            )

        group_results.append(
            {
                "name": group_name,
                "count": len(selected),
                "top_k": top_k,
                "neighbors": [
                    {
                        "name": n,
                        "label": labels[n],
                        "items": [
                            {"name": other, "label": labels[other], "sim": score}
                            for other, score in neighbors.get(n, [])
                        ],
                    }
                    for n in sorted_names
                ],
                "adjacent_avg": adjacent_avg,
                "far_avg": far_avg,
                "gap": gap,
                "top1": top1_ok,
                "top3": top3_ok,
                "evaluated": total,
            }
        )

    return {
        "pairs": pair_results,
        "groups": group_results,
        "report": "\n".join(report_lines).strip() + "\n",
        "space": {
            "expected_space_id": expected_space_id,
            "enforce_single_space": enforce_single_space,
            "found_space_ids": space_counts,
            "missing_space_id": missing_space_id,
            "excluded": excluded,
            "included": len(rows),
            "total": len(input_rows),
        }
        if expected_space_id or space_counts or missing_space_id or excluded
        else None,
    }
