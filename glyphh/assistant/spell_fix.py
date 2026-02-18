"""
Offline spell correction using Levenshtein distance against known vocabulary.

Zero external dependencies — pure Python standard library only.
"""


class OfflineSpellFixer:
    """Levenshtein-based spell correction against known vocabulary."""

    def __init__(
        self,
        known_objects: list[str],
        verb_map: dict[str, str],
        exemplar_questions: list[str],
        common_misspellings: dict[str, str] | None = None,
    ):
        """Build vocabulary from known objects, verb map keys, and exemplar questions."""
        self._known_objects = {w.lower() for w in known_objects}
        self._verb_map = verb_map
        self._exemplar_questions = exemplar_questions
        self._common_misspellings: dict[str, str] = {
            k.lower(): v.lower() for k, v in (common_misspellings or {}).items()
        }
        self._vocabulary: set[str] = self._build_vocabulary()

    def _build_vocabulary(self) -> set[str]:
        """Combine all sources into a flat set of lowercased words."""
        vocab: set[str] = set()
        # Source 1: known objects
        vocab.update(self._known_objects)
        # Source 2: verb map keys
        vocab.update(k.lower() for k in self._verb_map)
        # Source 3: unique words from exemplar questions
        for question in self._exemplar_questions:
            vocab.update(w.lower() for w in question.split())
        return vocab

    @staticmethod
    def _levenshtein_distance(a: str, b: str) -> int:
        """Standard dynamic-programming Levenshtein distance. Pure Python."""
        m, n = len(a), len(b)
        # Use a single-row DP for space efficiency
        prev = list(range(n + 1))
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            curr[0] = i
            for j in range(1, n + 1):
                cost = 0 if a[i - 1] == b[j - 1] else 1
                curr[j] = min(
                    prev[j] + 1,       # deletion
                    curr[j - 1] + 1,   # insertion
                    prev[j - 1] + cost, # substitution
                )
            prev, curr = curr, prev
        return prev[n]

    def _best_match(self, word: str) -> str:
        """Find the closest vocabulary word within distance 2.

        Prefers ``_known_objects`` on ties.  Returns the original *word*
        unchanged when no vocabulary entry is within distance 2.
        """
        lower_word = word.lower()
        if lower_word in self._vocabulary:
            return lower_word

        best_word = lower_word
        best_dist = 3  # anything > 2 means "no match found"

        for vocab_word in self._vocabulary:
            dist = self._levenshtein_distance(lower_word, vocab_word)
            if dist > best_dist:
                continue
            if dist < best_dist:
                best_dist = dist
                best_word = vocab_word
            elif dist == best_dist:
                # Tie-break: prefer known objects
                current_is_known = best_word in self._known_objects
                candidate_is_known = vocab_word in self._known_objects
                if candidate_is_known and not current_is_known:
                    best_word = vocab_word

        return best_word

    def fix(self, text: str) -> tuple[str, bool]:
        """Correct misspellings in *text*.

        Returns ``(corrected_text, was_changed)``.
        """
        tokens = text.split()
        corrected: list[str] = []

        for token in tokens:
            lower_token = token.lower()
            # 1. Check common misspellings map first
            if lower_token in self._common_misspellings:
                corrected.append(self._common_misspellings[lower_token])
                continue
            # 2. Levenshtein-based match
            corrected.append(self._best_match(token))

        result = " ".join(corrected)
        changed = result != text
        return result, changed
