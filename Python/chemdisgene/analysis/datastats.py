"""
Stats on data
"""

import gzip
from collections import defaultdict, Counter

from ..data.pubtator import BinaryRelationship, parse_pubtator_to_dict, parse_relationships_opened_file


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

ENTITY_TYPES = ["Chemical", "Disease", "Gene"]

FOR_LATEX = False


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def get_counts(pbtr_file: str, ctd_relns_file: str, new_relns_file: str = None):
    docs_dict = parse_pubtator_to_dict(pbtr_file, ctd_relns_file)

    if new_relns_file:
        if new_relns_file.endswith(".gz"):
            with gzip.open(new_relns_file) as f:
                relationships = parse_relationships_opened_file(f, from_ctd=False)
        else:
            with open(new_relns_file) as f:
                relationships = parse_relationships_opened_file(f, from_ctd=False)

        for docid, relns in relationships.items():
            doc = docs_dict[docid]
            for reln in relns:
                try:
                    doc.add_relationship(reln)
                except AssertionError as e:
                    print(e)
                    print("Skipping entry ...\n")

    counts = dict(n_docs=len(docs_dict),
                  n_docs_no_relns=0,
                  n_docs_no_ctd_relns=0,

                  n_mentions=Counter(),
                  n_linked_mentions=Counter(),
                  n_unlinked_mentions=Counter(),

                  n_unique_mentions=Counter(),
                  n_unique_linked_mentions=Counter(),

                  n_relns=Counter(),
                  n_ctd_relns=Counter(),
                  n_non_ctd_relns=Counter(),

                  n_unique_relns=Counter(),

                  n_unique_entities_in_relns=Counter(),
                  )

    n_docs_no_relns = 0
    n_docs_no_ctd_relns = 0
    unique_mentioned_entities = set()
    unique_relns = set()

    for doc in docs_dict.values():
        for men in doc.mentions:
            counts["n_mentions"][men.entity_type] += 1
            if men.is_unresolved_mention():
                counts["n_unlinked_mentions"][men.entity_type] += 1
            else:
                counts["n_linked_mentions"][men.entity_type] += 1

        unique_mentioned_entities |= doc.get_mentioned_entities()

        for reln in doc.relationships:
            r_label = reln.get_pretty_relation_label()
            counts["n_relns"][r_label] += 1
            if reln.from_ctd:
                counts["n_ctd_relns"][r_label] += 1
            else:
                counts["n_non_ctd_relns"][r_label] += 1

        if len(doc.relationships) == 0:
            n_docs_no_relns += 1
            n_docs_no_ctd_relns += 1
        else:
            if not any(r.from_ctd for r in doc.relationships):
                n_docs_no_ctd_relns += 1

            unique_relns |= set(get_reln_key(reln) for reln in doc.relationships)

    counts["n_docs_no_relns"] = n_docs_no_relns
    counts["n_docs_no_ctd_relns"] = n_docs_no_ctd_relns

    for etype, eid in unique_mentioned_entities:
        counts["n_unique_mentions"][etype] += 1
        if eid != "-":
            counts["n_unique_linked_mentions"][etype] += 1

    unique_entities_in_relns = defaultdict(set)
    for reln_key in unique_relns:
        counts["n_unique_relns"][get_pretty_reln_label(reln_key)] += 1
        unique_entities_in_relns[reln_key[1]].add((reln_key[2]))
        unique_entities_in_relns[reln_key[3]].add((reln_key[4]))

    for k, v in unique_entities_in_relns.items():
        counts["n_unique_entities_in_relns"][k] = len(v)

    return counts


def get_reln_key(reln: BinaryRelationship):
    return reln.relation_label, reln.subj_type, reln.subj_eid, reln.obj_type, reln.obj_eid


def get_pretty_reln_label(reln_key):
    return BinaryRelationship(reln_key[2], reln_key[3], reln_key[0]).get_pretty_relation_label()


def print_stats_ctd_derived(data_dir="../data/ctd_derived"):

    stats = dict()
    all_splits = ["train", "dev", "test"]
    for split in all_splits:
        print(f"Processing {data_dir}/{split} ...", flush=True)
        stats[split] = get_counts(f"{data_dir}/{split}_abstracts.txt.gz", f"{data_dir}/{split}_relationships.tsv.gz")

    print()
    print()
    print("Basic Stats")
    print("===========")
    print()

    pp_counts("Stat", all_splits)
    pp_counts("-" * 35, ["-" * 9 for _ in all_splits])

    pp_counts("Nbr. abstracts", [stats[split]["n_docs"] for split in all_splits])
    pp_counts("... with no relationships", [stats[split]["n_docs_no_relns"] for split in all_splits])

    for split in all_splits:
        for k in list(stats[split].keys()):
            if not isinstance(stats[split][k], int):
                stats[split][k]["Total"] = sum(stats[split][k].values())

    pp_counts("Nbr. relationships", [stats[split]["n_relns"]["Total"] for split in all_splits])
    pp_counts("... unique relationships", [stats[split]["n_unique_relns"]["Total"] for split in all_splits])

    pp_counts("Total Entity mentions", [stats[split]["n_mentions"]["Total"] for split in all_splits])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", [stats[split]["n_mentions"][etype] for split in all_splits])

    pp_counts("Total Linked Entity mentions", [stats[split]["n_linked_mentions"]["Total"] for split in all_splits])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", [stats[split]["n_linked_mentions"][etype] for split in all_splits])

    pp_counts("Total Unlinked Entity mentions", [stats[split]["n_unlinked_mentions"]["Total"] for split in all_splits])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", [stats[split]["n_unlinked_mentions"][etype] for split in all_splits])

    pp_counts("Total Unique Entity mentions", [stats[split]["n_unique_mentions"]["Total"] for split in all_splits])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", [stats[split]["n_unique_mentions"][etype] for split in all_splits])

    pp_counts("Unique Unlinked Entity mentions",
              [stats[split]["n_unique_linked_mentions"]["Total"] for split in all_splits])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", [stats[split]["n_unique_linked_mentions"][etype] for split in all_splits])

    pp_counts("Unique Entities in relns.",
              [stats[split]["n_unique_entities_in_relns"]["Total"] for split in all_splits])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", [stats[split]["n_unique_entities_in_relns"][etype] for split in all_splits])

    print("-" * 71)
    print()
    print()

    print("Distribution of Relation Types")
    print("==============================")
    print()
    pp_counts("", ["", "Total", "", "", "Unique", ""], label_width=50)
    pp_counts("", ["-" * 33] * 2, label_width=50)
    pp_counts("Relation Type", all_splits * 2, label_width=50)
    pp_counts("-" * 50, ["-" * 9 for _ in all_splits] * 2, label_width=50)

    for rtype in sorted(stats["train"]["n_relns"].keys()):
        pp_counts(rtype,
                  [stats[split]["n_relns"][rtype] for split in all_splits]
                  + [stats[split]["n_unique_relns"][rtype] for split in all_splits],
                  label_width=50)

    print("-" * 122)
    print()
    return


def pp_counts(label, counts, label_width=35):
    if FOR_LATEX:
        sep = " & "
        end = " \\\\\n"
    else:
        sep = "   "
        end = "\n"

    if isinstance(counts, (str, int)):
        counts = [counts]

    counts = [f"{c:9,d}" if isinstance(c, int) else f"{c:>9s}" for c in counts]
    print(f"{label:{label_width}s}", *counts, sep=sep, end=end)
    return


def print_stats_curated(data_dir="../data/curated"):

    print(f"Processing {data_dir} ...", flush=True)

    stats = get_counts(f"{data_dir}/abstracts.txt.gz", f"{data_dir}/approved_relns_ctd_v0.tsv.gz",
                       f"{data_dir}/approved_relns_new_v0.tsv.gz")

    print()
    print()
    print("Basic Stats")
    print("===========")
    print()

    pp_counts("Stat", "Count")
    pp_counts("-" * 35, "-" * 9)

    pp_counts("Nbr. abstracts", stats["n_docs"])
    pp_counts("... with no relationships", stats["n_docs_no_relns"])
    pp_counts("... with no CTD relationships", stats["n_docs_no_ctd_relns"])

    for k in list(stats.keys()):
        if not isinstance(stats[k], int):
            stats[k]["Total"] = sum(stats[k].values())

    pp_counts("Nbr. relationships", stats["n_relns"]["Total"])
    pp_counts("... unique relationships", stats["n_unique_relns"]["Total"])
    pp_counts("Nbr. CTD relationships", stats["n_ctd_relns"]["Total"])
    pp_counts("Nbr. New relationships", stats["n_non_ctd_relns"]["Total"])

    pp_counts("Total Entity mentions", stats["n_mentions"]["Total"])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", stats["n_mentions"][etype])

    pp_counts("Total Linked Entity mentions", stats["n_linked_mentions"]["Total"])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", stats["n_linked_mentions"][etype])

    pp_counts("Total Unlinked Entity mentions", stats["n_unlinked_mentions"]["Total"])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", stats["n_unlinked_mentions"][etype])

    pp_counts("Total Unique Entity mentions", stats["n_unique_mentions"]["Total"])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", stats["n_unique_mentions"][etype])

    pp_counts("Unique Unlinked Entity mentions", stats["n_unique_linked_mentions"]["Total"])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", stats["n_unique_linked_mentions"][etype])

    pp_counts("Unique Entities in relns.", stats["n_unique_entities_in_relns"]["Total"])
    for etype in ENTITY_TYPES:
        pp_counts(f"    {etype}s", stats["n_unique_entities_in_relns"][etype])

    print("-" * 47)
    print()
    print()

    print("Distribution of Relation Types")
    print("==============================")
    print()
    pp_counts("Relation Type", ["New", "CTD"], label_width=50)
    pp_counts("-" * 50, ["-" * 9] * 2, label_width=50)

    for rtype in sorted(stats["n_relns"].keys()):
        if stats["n_non_ctd_relns"][rtype]:
            non_ctd_pct = "{:.1%}".format(stats["n_non_ctd_relns"][rtype] / stats["n_non_ctd_relns"]["Total"])
        else:
            non_ctd_pct = ""

        if stats["n_ctd_relns"][rtype]:
            ctd_pct = "{:.1%}".format(stats["n_ctd_relns"][rtype] / stats["n_ctd_relns"]["Total"])
        else:
            ctd_pct = ""

        pp_counts(rtype, [non_ctd_pct, ctd_pct], label_width=50)

    print("-" * 74)
    print()
    return


# ======================================================================================================
#   Main
# ======================================================================================================

# Invoke as: python -m chemdisgene.analysis.datastats CMD ...
# e.g.
# python -m chemdisgene.analysis.datastats ctd_basic ../data/ctd_derived

if __name__ == '__main__':

    import argparse
    from datetime import datetime

    _argparser = argparse.ArgumentParser(
        description='Corpus statistics.',
    )

    _subparsers = _argparser.add_subparsers(dest='subcmd',
                                            title='Available commands',
                                            )
    # Make the sub-commands required
    _subparsers.required = True

    # ... ctd_basic DATA_DIR
    _sub_cmd_parser = _subparsers.add_parser('ctd_basic',
                                             help="Basic statistics on CTD-derived Corpus.")

    _sub_cmd_parser.add_argument('data_dir', type=str,
                                 help="Path to `ChemDisGene/data/ctd_derived` dir, e.g. `../data/ctd_derived`.")

    # ... curated_basic DATA_DIR
    _sub_cmd_parser = _subparsers.add_parser('curated_basic',
                                             help="Basic statistics on CTD-derived Corpus.")

    _sub_cmd_parser.add_argument('data_dir', type=str,
                                 help="Path to `ChemDisGene/data/curated` dir, e.g. `../data/curated`.")

    # --

    _args = _argparser.parse_args()
    # .................................................................................................

    start_time = datetime.now()

    print()

    if _args.subcmd == 'ctd_basic':

        print_stats_ctd_derived(_args.data_dir)

    elif _args.subcmd == 'curated_basic':

        print_stats_curated(_args.data_dir)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)