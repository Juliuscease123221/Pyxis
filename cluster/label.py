"""Phase 3 step 4: label every cluster node with sibling-scoped c-TF-IDF.

### Why the scope is the whole trick

c-TF-IDF concatenates each cluster's members into one meta-document and scores
terms by in-cluster frequency against cross-cluster frequency. The question is
what "cross-cluster" ranges over.

Scored against the **global** corpus, a term is distinctive if it is rare in the
catalog. `Action` is on 24,864 of 56,129 games, so it is never rare, so it never
scores -- but neither does anything else inside an action cluster, because the
whole subtree shares the same vocabulary. Every child of "Action" ends up
labelled by whatever incidental term happens to be globally uncommon, or by
"Action" again, and the path reads Action -> Action -> Action Indie.

Scored against **siblings**, the question changes from "what is unusual in the
catalog" to "what distinguishes this child from the other children of the same
parent". Inside Action, `Action` is on ~100% of every sibling, so its sibling-df
is maximal and it scores zero *automatically*. `Metroidvania` is on one sibling
and not the others, so it wins. The generic term suppresses itself.

This is why there is no stopword list here, and why there must not be one. A
hand-maintained list of "too common" tags would be a workaround for scoping the
denominator wrongly; with the denominator right, commonness is measured locally
and handled by the mathematics. If a generic term ever dominates a label, the
bug is in the scope, not in the vocabulary.

### Term weighting

A game's tags are vote-ranked, so tag 1 is a stronger claim than tag 15.
Contribution decays as 1/(1+rank), matching variant B's construction.

Usage:
    python -m cluster.label --tree tree_main
    python -m cluster.label --tree tree_main --top 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from embed.common import load_games

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cluster"


def subtree_members(nodes: dict, node: str) -> list[int]:
    """Every game under `node`, including its own directly-held members.

    A parent is labelled from its whole subtree, not just the residue that fell
    out at its own level -- otherwise an internal node is described by its
    leftovers rather than by what it contains.
    """
    out: list[int] = []
    stack = [node]
    while stack:
        n = stack.pop()
        rec = nodes[str(n)]
        out.extend(rec["members"])
        stack.extend(rec["children"])
    return out


def term_weights(g, members: list[int], max_tags: int = 20) -> dict[str, float]:
    """Rank-decayed tag mass for a cluster's meta-document."""
    w: dict[str, float] = defaultdict(float)
    for i in members:
        for r, t in enumerate(g.tags[i][:max_tags]):
            w[t] += 1.0 / (1.0 + r)
    return dict(w)


def term_coverage(g, members: list[int], max_tags: int = 20) -> dict[str, float]:
    """Fraction of a cluster's games carrying each tag.

    Coverage, not summed weight, is what makes a label *true of the cluster*.
    A tag on 2 of 500 games can dominate a frequency-normalised score once the
    genuinely common tags are suppressed; coverage makes that impossible.
    """
    n = max(len(members), 1)
    c: dict[str, float] = defaultdict(float)
    for i in members:
        for t in g.tags[i][:max_tags]:
            c[t] += 1.0
    return {t: v / n for t, v in c.items()}


def ctfidf_against_siblings(docs: dict[str, dict[str, float]], top: int
                            ) -> dict[str, list[tuple[str, float]]]:
    """c-TF-IDF where the corpus is exactly the sibling set.

        tf(t, c)  = weight of t in c / total weight in c
        idf(t)    = log(n_siblings / df(t))          df over siblings only
        score     = tf * idf

    **Document frequency is the wrong denominator here, in both directions.**

    Two earlier attempts failed in opposite ways, and the pair is instructive:

    * `tf * log(1 + n/df)` (BERTopic's smoothed form). A tag on every sibling
      still scores `log(2) = 0.69`, and generic tags also carry the highest tf,
      so they win. 90% of children reused a top-3 term from their parent -- the
      "Action -> Action -> Action Indie" failure verbatim.
    * `tf * log(n/df)`. Now a tag on every sibling scores exactly 0, which kills
      generic terms -- but it also kills the *defining* term of a coherent
      cluster, because every child of a farming-sim cluster carries "Farming
      Sim". With the real terms zeroed, whatever tag happens to sit in exactly
      one sibling wins, however rare. Labels came out "Memes / Naval Combat /
      Pirates" for the Stardew Valley neighbourhood.

    Both failures come from `df` being a *binary* presence count: it cannot
    tell "on 90% of every sibling" from "on 2% of every sibling".

    **Prevalence x lift.** Using coverage -- the fraction of a cluster's games
    carrying the tag -- on both sides fixes it:

        p_c(t)    = coverage of t inside this cluster
        p_rest(t) = coverage of t across the sibling set, excluding this cluster
        score     = p_c(t) * log( (p_c(t) + eps) / (p_rest(t) + eps) )

    The `log` ratio is the distinctiveness and the leading `p_c` is the
    prevalence, so a term must be both common *here* and less common *there*.

      * on 90% here and 85% of siblings -> near-zero log, suppressed
      * on 2% here and 0% elsewhere     -> large log, but p_c kills it
      * on 70% here and 10% elsewhere   -> wins, correctly

    A tag that is genuinely defining survives even when every sibling has some
    of it, because the ratio is between *degrees*, not between presence and
    absence. This is still sibling scoping -- the denominator is the sibling
    set, never the global corpus -- and there is still no stopword list.
    """
    n = len(docs)
    eps = 1e-3

    totals: dict[str, float] = defaultdict(float)
    for w in docs.values():
        for t, v in w.items():
            totals[t] += v

    out: dict[str, list[tuple[str, float]]] = {}
    for cid, w in docs.items():
        if n > 1:
            scored = []
            for t, p_c in w.items():
                # mean coverage across the other siblings
                p_rest = (totals[t] - p_c) / (n - 1)
                scored.append((t, p_c * math.log((p_c + eps) / (p_rest + eps))))
        else:
            scored = list(w.items())
        scored.sort(key=lambda kv: -kv[1])
        out[cid] = scored[:top]
    return out


def phrase(terms: list[tuple[str, float]], k: int = 3) -> str:
    """Join the top terms into something readable.

    Deliberately simple: the raw c-TF-IDF terms are the fallback SPEC.md
    specifies, and LLM tidying is the first item on the cut list.
    """
    picked = [t for t, s in terms[:k] if s > 0]
    return " / ".join(picked) if picked else "unnamed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tree", default="tree_main")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--phrase-terms", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    g = load_games()
    path = DATA / f"{args.tree}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    nodes = payload["nodes"]
    root = str(payload["root"])

    # group nodes by parent: each group is one sibling set, one c-TF-IDF corpus
    groups: dict[str, list[str]] = defaultdict(list)
    for nid, rec in nodes.items():
        if rec["parent"] is None:
            continue
        groups[str(rec["parent"])].append(nid)

    print(f"tree {args.tree}: {len(nodes):,} nodes, {len(groups):,} sibling sets")

    # coverage, not summed weight: see ctfidf_against_siblings
    cache: dict[str, dict[str, float]] = {}
    for nid in nodes:
        cache[nid] = term_coverage(g, subtree_members(nodes, nid))

    labelled = 0
    for parent, sibs in groups.items():
        docs = {s: cache[s] for s in sibs}
        scored = ctfidf_against_siblings(docs, args.top)
        for s, terms in scored.items():
            nodes[s]["terms"] = [[t, round(v, 6)] for t, v in terms]
            nodes[s]["label"] = phrase(terms, args.phrase_terms)
            labelled += 1

    nodes[root]["label"] = "All games"
    nodes[root]["terms"] = []

    out = Path(args.out) if args.out else DATA / f"{args.tree}_labelled.json"
    out.write_text(json.dumps(payload), encoding="utf-8")
    print(f"labelled {labelled:,} nodes -> {out}")

    # how often does a child repeat a term from its parent's label?
    repeats = 0
    checked = 0
    for nid, rec in nodes.items():
        p = rec["parent"]
        if p is None or "label" not in rec:
            continue
        pl = nodes[str(p)].get("label", "")
        if not pl or pl == "All games":
            continue
        checked += 1
        child_terms = {t for t, _ in rec.get("terms", [])[:3]}
        parent_terms = {t for t, _ in nodes[str(p)].get("terms", [])[:3]}
        if child_terms & parent_terms:
            repeats += 1
    if checked:
        print(f"parent-term repetition: {repeats:,}/{checked:,} "
              f"({repeats / checked * 100:.1f}%) of children reuse a top-3 term "
              f"from their parent")
        print("  (high values mean the sibling scoping is not working)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
