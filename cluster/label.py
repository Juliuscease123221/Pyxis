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


# A residue bucket is a node whose best available label does not actually
# distinguish it. The motivating case: a 5,522-game node (10% of the catalog)
# labelled "Indie / Casual / VR", which is not a genre and which a viewer
# notices immediately.
#
# **Lift, not coverage, is the signal.** That node's top term covers 94% of its
# members -- a coverage test passes it easily -- but its lift over the sibling
# average is 1.68, the lowest of the twelve top-level territories by a wide
# margin. Every genuine territory scores far higher:
#
#     Action Roguelike / Bullet Hell    lift 60.9
#     Visual Novel / Anime / Romance    lift 31.6
#     Platformer / 2D Platformer        lift 26.3
#     Simulation / Management           lift  5.4
#     Indie / Casual / VR               lift  1.68   <- residue
#
# Across all nodes the lift distribution runs p10=2.07, p25=3.17, p50=5.61, so
# a threshold of 2.0 flags roughly the bottom decile. This is not a stopword
# list in disguise: nothing here names a tag. The test is structural -- "is
# this label substantially more true here than next door" -- and a globally
# ubiquitous tag fails it precisely because being everywhere is what makes it
# uninformative.
#
# Coverage is kept only as a weak floor, for a label true of almost nobody.
RESIDUE_MIN_LIFT = 2.0        # top term must be 2x commoner here than in siblings
RESIDUE_MIN_COVERAGE = 0.25   # ... and describe at least a quarter of members


def label_quality(coverage: dict[str, float], sibling_mean: dict[str, float],
                  terms: list[tuple[str, float]]) -> dict:
    """How much of the cluster does its own label actually describe?

    Returns the top term's coverage and its lift over the sibling average, plus
    an `is_residue` flag. A residue node is one whose best available label is
    true of a minority of its members -- a catch-all, not a genre.
    """
    if not terms:
        return {"top_coverage": 0.0, "top_lift": 0.0, "is_residue": True}
    top = terms[0][0]
    p_c = coverage.get(top, 0.0)
    p_rest = sibling_mean.get(top, 0.0)
    lift = p_c / p_rest if p_rest > 1e-9 else float("inf")
    return {
        "top_coverage": round(p_c, 4),
        "top_lift": round(min(lift, 999.0), 3),
        "is_residue": bool(p_c < RESIDUE_MIN_COVERAGE or lift < RESIDUE_MIN_LIFT),
    }


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
    residue = 0
    for parent, sibs in groups.items():
        docs = {s: cache[s] for s in sibs}
        scored = ctfidf_against_siblings(docs, args.top)
        # mean coverage across the sibling set, for the lift in label_quality
        totals: dict[str, float] = defaultdict(float)
        for w in docs.values():
            for t, v in w.items():
                totals[t] += v
        for s, terms in scored.items():
            others = len(sibs) - 1
            sib_mean = {
                t: ((totals[t] - docs[s].get(t, 0.0)) / others) if others else 0.0
                for t, _ in terms
            }
            q = label_quality(docs[s], sib_mean, terms)
            nodes[s]["terms"] = [[t, round(v, 6)] for t, v in terms]
            nodes[s]["label"] = phrase(terms, args.phrase_terms)
            nodes[s].update(q)
            labelled += 1
            residue += q["is_residue"]

    nodes[root]["label"] = "All games"
    nodes[root]["terms"] = []
    nodes[root]["is_residue"] = False

    print(f"residue nodes: {residue:,}/{labelled:,} "
          f"({residue / max(labelled, 1) * 100:.1f}%) -- top term lift"
          f" <{RESIDUE_MIN_LIFT} over siblings, or coverage"
          f" <{RESIDUE_MIN_COVERAGE:.0%} of members")
    big_res = sorted(
        (nodes[n] for n in nodes if nodes[n].get("is_residue")),
        key=lambda r: -r.get("size", 0))[:5]
    if big_res:
        print("  largest residue nodes (render as unclustered, not as a genre):")
        for r in big_res:
            print(f"    {r.get('size', 0):>7,}  {r.get('label', '?')[:48]:<48}"
                  f"  top-term coverage {r.get('top_coverage', 0) * 100:.0f}%")

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
