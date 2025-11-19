import re
from typing import Iterable, List, Union, Optional, Any

ProofItem = Union[str, Iterable["ProofItem"]]  # nested lists possible in your data

_THEOREM_HEADER_RE = re.compile(r"^\s*theorem\s+")
_APPLY_RE          = re.compile(r"^\s*apply\s+(.+?)\.\s*$", re.IGNORECASE)
_INTRO_RE          = re.compile(r"^\s*intro\b", re.IGNORECASE)
_SHOW_BY_RE        = re.compile(r"^\s*show\s+(?P<goal>.+?)\s+by\s+(?P<tactic>[^.]+)\.\s*$", re.IGNORECASE)

def _flatten(proof: Iterable[ProofItem]) -> List[str]:
    """Flatten a 'proof' that may contain nested lists/blocks into a simple list of lines."""
    out: List[str] = []
    stack: List[Iterable[ProofItem]] = [proof]
    while stack:
        top = stack.pop()
        if top is None:
            continue
        if isinstance(top, (list, tuple)):
            # push in reverse to preserve order
            for item in reversed(list(top)):
                stack.append(item)
        else:
            s = str(top)
            # break multi-line strings into lines
            for line in s.splitlines():
                out.append(line.rstrip())
    return out

def _strip_block_syntax(s: str) -> Optional[str]:
    """Return None for braces/goal headers; otherwise the trimmed content."""
    t = s.strip()
    if not t:
        return None
    if t == "}":
        return None
    if t.endswith("{"):
        # lines like "goal ... {" or the theorem header line with "{"
        if t.lower().startswith("goal "):
            return None
        if _THEOREM_HEADER_RE.match(t):
            return None
    # pure theorem header without "{"
    if _THEOREM_HEADER_RE.match(t):
        return None
    return t

def _infer_em_hint(goal: str, problem: Optional[str]) -> str:
    """
    Best-effort inference to match your examples.
    Returns a string that already includes surrounding parentheses and a trailing period.
    """
    g = goal.replace("  ", " ").lower()

    # Easy wins
    if "(or false" in g or " or false)" in g:
        return "(em false)."

    # If the goal is an 'or' of a function type and its negation: (or [f] (not [f]))
    if g.startswith("(or [") and " (not [" in g:
        # Heuristic: look inside the function's codomain to pick a constructor name
        codomain = None
        # try to grab the last '-> something)' bit
        m = re.search(r"->\s*\(([^()]+)\)\]\)", g)
        if m:
            codomain = m.group(1)
        # fallbacks based on common connectives
        if codomain:
            if codomain.startswith("and "):   return "(em and_i@type)."
            if codomain.startswith("iff "):   return "(em iff_i@type)."
            if codomain.startswith("or "):    return "(em or_il@type)."
            if codomain.startswith("not "):   return "(em not_i@type)."
        # last-resort for function types: exfalso often appears for false → …
        if "false ->" in g:
            return "(em exfalso@type)."
        return "(em or_il@type)."

    # Plain OR goals (law of excluded middle on a proposition)
    if g.startswith("(or "):
        # Try to detect connective in the left disjunct to choose a type-level intro
        left = None
        depth = 0
        buf = []
        # crude parser to take the first top-level token after "(or "
        src = g[len("(or "):]
        for ch in src:
            if ch == "(":
                depth += 1
            elif ch == ")":
                if depth == 0:
                    break
                depth -= 1
            if depth < 0:
                break
            buf.append(ch)
            # stop when we've likely closed the left arg
            if depth == 0 and ch == ")":
                break
        left = "".join(buf).strip().lstrip("(")
        if left:
            if left.startswith("and "): return "(em and_i@type)."
            if left.startswith("not "): return "(em not_i@type)."
            if left.startswith("iff "): return "(em iff_i@type)."
            if left.startswith("or "):  return "(em or_il@type)."
        # If nothing matched, pick a reasonable default
        return "(em or_il@type)."

    # If the goal looks like a negation
    if g.startswith("(not "):
        return "(em not_i@type)."

    # If we see an antecedent 'false -> ...', exfalso is a good bet
    if "false ->" in g:
        return "(em exfalso@type)."

    # Fallback
    return "(em or_il@type)."

def convert_proof_to_actions_prop(proof: Iterable[ProofItem],
                             problem: Optional[str] = None) -> List[str]:
    """
    Convert the given 'proof' list (strings and possibly nested lists/blocks) into the target 'actions' list.
    You may pass the row's 'problem' string for slightly better EM-hint inference; it's optional.
    """
    actions: List[str] = []
    lines = _flatten(proof)

    for raw in lines:
        t = _strip_block_syntax(raw)
        if t is None:
            continue

        # intro
        if _INTRO_RE.match(t) and t.endswith("."):
            actions.append("intro.")
            continue

        # apply
        m_apply = _APPLY_RE.match(t)
        if m_apply:
            tactic = m_apply.group(1).strip()
            actions.append(f"a {tactic}")
            actions.append("=> .")
            continue

        # show ... by ...
        m_show = _SHOW_BY_RE.match(t)
        if m_show:
            goal = m_show.group("goal").strip()
            tactic = m_show.group("tactic").strip()
            actions.append(f"c {tactic}")
            if tactic.lower() == "em":
                hint = _infer_em_hint(goal, problem)
                actions.append(f"=> {hint}")
            else:
                # For non-EM tactics, we conservatively add a generic continuation.
                # You can specialize this if you want richer hints, e.g., for iff_er, and_el/er, or_e, etc.
                actions.append("=> .")
            continue

        # Everything else is ignored (headers already filtered; nested 'goal ... {' lines filtered)
        # If you prefer to be strict, raise on unknown lines instead of ignoring.
        # raise ValueError(f"Unrecognized proof line: {t}")

    return actions