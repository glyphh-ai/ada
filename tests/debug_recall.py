"""Debug: trace encoding and similarity for specific query/thought pairs."""

from glyphh.memory.thought_space import ThoughtGlyphSpace
from glyphh.memory.thought_glyph import ThoughtGlyphEncoder, ROLE_TO_LAYER_SEGMENT
from glyphh.core.ops import cosine_similarity


def trace_encoding(encoder: ThoughtGlyphEncoder, text: str, speaker: str = "incoming"):
    """Show exactly how a text gets encoded."""
    words = encoder._tokenize(text)
    print(f"\n  Text: \"{text}\"")
    print(f"  Tokens: {words}")

    segment_roles = {}
    content_words = []

    for word in words:
        matched = encoder._primitives.match_roles(word)
        if matched:
            roles = [(r, f"{s:.3f}") for r, s in matched]
            print(f"    \"{word}\" → primitives: {roles}")
            for role, _score in matched:
                if role in ROLE_TO_LAYER_SEGMENT:
                    layer, seg = ROLE_TO_LAYER_SEGMENT[role]
                    key = f"{layer}_{seg}"
                    if key not in segment_roles:
                        segment_roles[key] = role
        else:
            print(f"    \"{word}\" → CONTENT")
            content_words.append(word)

    print(f"  Content words: {content_words}")
    print(f"  Activated segments: {sorted(segment_roles.keys())}")
    for key, role in sorted(segment_roles.items()):
        value = " ".join([role] + content_words)
        print(f"    {key} = \"{value}\"")


def trace_similarity(space: ThoughtGlyphSpace, query: str, targets: list[str]):
    """Show detailed similarity between query and specific stored thoughts."""
    query_glyph = space.encoder.encode_thought(query, speaker="incoming")
    query_activated = set(query_glyph.metadata.get("_activated_attrs", []))

    print(f"\n  Query: \"{query}\"")
    print(f"  Query activated: {sorted(query_activated)}")

    for target in targets:
        # Find stored thought by content
        stored = None
        for t in space._thoughts.values():
            if t.content == target:
                stored = t
                break
        if not stored:
            print(f"\n  Target \"{target}\" not found!")
            continue

        stored_activated = set(stored.glyph.metadata.get("_activated_attrs", []))
        global_sim = float(cosine_similarity(
            query_glyph.global_cortex.data,
            stored.glyph.global_cortex.data,
        ))

        overlap = query_activated & stored_activated
        union = query_activated | stored_activated
        structural = len(overlap) / len(union) if union else 0

        print(f"\n  vs \"{target}\"")
        print(f"    Stored activated: {sorted(stored_activated)}")
        print(f"    Overlap: {sorted(overlap)} ({len(overlap)}/{len(union)} = {structural:.3f})")
        print(f"    Global cortex cos: {global_sim:.4f}")

        # Role-level per segment
        _LAYER_WEIGHTS = {
            "perspective": 0.25, "semantic": 0.30,
            "relational": 0.25, "temporal": 0.10, "direction": 0.10,
        }
        for layer_name, query_layer in query_glyph.layers.items():
            if layer_name not in stored.glyph.layers:
                continue
            stored_layer = stored.glyph.layers[layer_name]
            for seg_name, query_seg in query_layer.segments.items():
                attr_key = f"{layer_name}_{seg_name}"
                if attr_key not in query_activated:
                    continue
                if attr_key not in stored_activated:
                    print(f"    {attr_key}: MISSING in stored (→ 0)")
                    continue
                if seg_name not in stored_layer.segments:
                    continue
                stored_seg = stored_layer.segments[seg_name]
                for rn, qv in query_seg.roles.items():
                    if rn in stored_seg.roles:
                        sv = stored_seg.roles[rn]
                        rsim = float(cosine_similarity(qv.data, sv.data))
                        print(f"    {attr_key}: cos = {rsim:.4f}")


def main():
    space = ThoughtGlyphSpace()

    # Absorb key thoughts
    thoughts = [
        ("my name is chris", "incoming"),
        ("i am chris", "incoming"),
        ("my wife is named sarah", "incoming"),
        ("my son is named jake", "incoming"),
        ("my daughter is named emma", "incoming"),
        ("i am tired today", "incoming"),
        ("i like pizza", "incoming"),
        ("i work as a software engineer", "incoming"),
        ("we live in austin texas", "incoming"),
        ("jake loves minecraft", "incoming"),
        ("emma likes drawing", "incoming"),
        ("i feel happy when i code", "incoming"),
    ]
    for text, spk in thoughts:
        space.absorb(text, speaker=spk)

    print("=" * 70)
    print("  ENCODING TRACES")
    print("=" * 70)

    # Trace encodings for key texts
    enc = space.encoder
    for text, _ in thoughts[:6]:
        trace_encoding(enc, text)

    # Trace query encodings
    queries = [
        "what is my name?",
        "who am i?",
        "what is my wife's name?",
        "what does jake like?",
    ]
    for q in queries:
        trace_encoding(enc, q)

    print("\n" + "=" * 70)
    print("  SIMILARITY TRACES")
    print("=" * 70)

    trace_similarity(space, "what is my name?", [
        "my name is chris",
        "i am chris",
        "my son is named jake",
        "i am tired today",
    ])

    trace_similarity(space, "what is my wife's name?", [
        "my wife is named sarah",
        "my name is chris",
        "i am chris",
    ])

    trace_similarity(space, "what does jake like?", [
        "jake loves minecraft",
        "i like pizza",
        "emma likes drawing",
    ])


if __name__ == "__main__":
    main()
