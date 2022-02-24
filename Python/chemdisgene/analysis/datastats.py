"""
Stats on data
"""

import gzip
from collections import defaultdict, Counter
import os.path
from typing import List, Union

from ..data.pubtator import BinaryRelationship, parse_pubtator_to_dict, parse_relationships_opened_file


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

ENTITY_TYPES = ["Chemical", "Disease", "Gene"]

FOR_LATEX = False


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------


def read_docs_relns(pbtr_file: str, ctd_relns_file: str, new_relns_file: str = None):
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

    return docs_dict


def get_counts(pbtr_file: str, ctd_relns_file: str, new_relns_file: str = None):
    docs_dict = read_docs_relns(pbtr_file, ctd_relns_file, new_relns_file)

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
            r_label = reln.relation_label
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
        counts["n_unique_relns"][reln_key[0]] += 1
        unique_entities_in_relns[reln_key[1]].add((reln_key[2]))
        unique_entities_in_relns[reln_key[3]].add((reln_key[4]))

    for k, v in unique_entities_in_relns.items():
        counts["n_unique_entities_in_relns"][k] = len(v)

    return counts


def get_reln_key(reln: BinaryRelationship):
    return reln.relation_label, reln.subj_type, reln.subj_eid, reln.obj_type, reln.obj_eid


def get_pretty_reln_label(relation_label):
    if relation_label == "Total":
        return relation_label
    else:
        return BinaryRelationship("Dummy", "Dummy", relation_label).get_pretty_relation_label()


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
    pp_counts(["", ""], ["", "Total", "", "", "Unique", ""], label_width=50)
    pp_counts(["", ""], ["-" * 33] * 2, label_width=50)
    pp_counts(["Relation Label", "Relation Type (pretty)"], all_splits * 2, label_width=50)
    pp_counts(["-" * 50] * 2, ["-" * 9 for _ in all_splits] * 2, label_width=50)

    for rtype in sorted(stats["train"]["n_relns"].keys(), key=get_pretty_reln_label):
        pp_counts([rtype, get_pretty_reln_label(rtype)],
                  [stats[split]["n_relns"][rtype] for split in all_splits]
                  + [stats[split]["n_unique_relns"][rtype] for split in all_splits],
                  label_width=50)

    print("-" * 175)
    print()
    return


def pp_counts(label: Union[str, List[str]], counts, label_width: Union[int, List[int]] = 35):
    if FOR_LATEX:
        sep = " & "
        end = " \\\\\n"
    else:
        sep = "   "
        end = "\n"

    if isinstance(counts, (str, int)):
        counts = [counts]

    if not isinstance(label, (list, tuple)):
        label = [label]
    if not isinstance(label_width, list):
        label_width = [label_width] * len(label)
    if len(label_width) < len(label):
        label_width += [label_width[-1]] * (len(label_width) - len(label))

    labels = [f"{lbl:{w}s}" for lbl, w in zip(label, label_width)]

    counts = [f"{c:9,d}" if isinstance(c, int) else f"{c:>9s}" for c in counts]
    print(*labels, *counts, sep=sep, end=end)
    return


def print_stats_curated(data_dir="../data/curated"):

    print(f"Processing {data_dir} ...", flush=True)

    ctd_relns_file = f"{data_dir}/approved_relns_ctd_v1.tsv.gz"
    new_relns_file = f"{data_dir}/approved_relns_new_v1.tsv.gz"
    if not os.path.exists(ctd_relns_file):
        ctd_relns_file = f"{data_dir}/approved_relns_ctd_v0.tsv.gz"
        new_relns_file = f"{data_dir}/approved_relns_new_v0.tsv.gz"

    stats = get_counts(f"{data_dir}/abstracts.txt.gz", ctd_relns_file, new_relns_file)

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
    pp_counts(["Relation Label", "Relation Type (pretty)"], ["New", "CTD"], label_width=50)
    pp_counts(["-" * 50] * 2, ["-" * 9] * 2, label_width=50)

    for rtype in sorted(stats["n_relns"].keys(), key=get_pretty_reln_label):
        if stats["n_non_ctd_relns"][rtype]:
            non_ctd_pct = "{:.1%}".format(stats["n_non_ctd_relns"][rtype] / stats["n_non_ctd_relns"]["Total"])
        else:
            non_ctd_pct = ""

        if stats["n_ctd_relns"][rtype]:
            ctd_pct = "{:.1%}".format(stats["n_ctd_relns"][rtype] / stats["n_ctd_relns"]["Total"])
        else:
            ctd_pct = ""

        pp_counts([rtype, get_pretty_reln_label(rtype)], [non_ctd_pct, ctd_pct], label_width=50)

    print("-" * 127)
    print()
    return


def pp_nrels_by_ndocs_curated(data_dir="../data/curated"):

    print(f"Processing {data_dir} ...", flush=True)

    ctd_relns_file = f"{data_dir}/approved_relns_ctd_v1.tsv.gz"
    new_relns_file = f"{data_dir}/approved_relns_new_v1.tsv.gz"
    if not os.path.exists(ctd_relns_file):
        ctd_relns_file = f"{data_dir}/approved_relns_ctd_v0.tsv.gz"
        new_relns_file = f"{data_dir}/approved_relns_new_v0.tsv.gz"

    docs_dict = read_docs_relns(f"{data_dir}/abstracts.txt.gz", ctd_relns_file, new_relns_file)

    all_reln_counts = Counter([len(doc.relationships)
                               for doc in docs_dict.values()])
    ctd_reln_counts = Counter([len([r for r in doc.relationships if r.from_ctd])
                               for doc in docs_dict.values()])
    new_reln_counts = Counter([len([r for r in doc.relationships if not r.from_ctd])
                               for doc in docs_dict.values()])

    max_n_relns = max(max(all_reln_counts.keys()), max(ctd_reln_counts.keys()))
    print("nRelns", "nDocs-All", "nDocs-CTD", "nDocs-New", sep="\t")
    for n_relns in range(max_n_relns + 1):
        print(n_relns,
              all_reln_counts[n_relns], ctd_reln_counts[n_relns], new_reln_counts[n_relns],
              sep="\t")
    print()

    return


# ======================================================================================================
#   Main
# ======================================================================================================

# Invoke as: python -m chemdisgene.analysis.datastats CMD ...
# e.g.
# python -m chemdisgene.analysis.datastats ctd_basic ../data/ctd_derived | tee ../data/ctd_derived/ctd_stats.txt
# python -m chemdisgene.analysis.datastats curated_basic ../data/curated | tee ../data/curated/curated_stats.txt
# python -m chemdisgene.analysis.datastats curated_distr ../data/curated

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

    # ... curated_distr DATA_DIR
    _sub_cmd_parser = _subparsers.add_parser('curated_distr',
                                             help="Distribution of nbr Relns by nbr Docs on CTD-derived Corpus.")

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

    elif _args.subcmd == 'curated_distr':

        pp_nrels_by_ndocs_curated(_args.data_dir)

    else:

        raise NotImplementedError(f"Command not implemented: {_args.subcmd}")

    # /

    print('\nTotal Run time =', datetime.now() - start_time)
