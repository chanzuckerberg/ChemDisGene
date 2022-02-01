"""
Reading and Writing PubTator format files.
"""

from collections import defaultdict
import gzip
import os
import re
import sys
from typing import Dict, List, Optional, Set, TextIO, Tuple


# -----------------------------------------------------------------------------
#   Globals
# -----------------------------------------------------------------------------

# Includes support for non-numeric Document IDs, e.g. "Manoj_Ramachandran_1|t|Fully Automating Graf’s Method for ..."
TITLE_ABSTR_PATT = re.compile(r'([^|]+)\|([ta])\|')


# -----------------------------------------------------------------------------
#   Classes
# -----------------------------------------------------------------------------

class BinaryRelationship:

    STDD_ARG_TYPES = {"chem": "Chemical",
                      "disease": "Disease",
                      "gene": "Gene"}

    def __init__(self, subj_eid: str, obj_eid: str, relation_label: str, from_ctd: bool = True):

        self.subj_eid = subj_eid
        self.obj_eid = obj_eid
        self.relation_label = relation_label
        self.from_ctd = from_ctd

        # Derived Fields:

        # Relation label contains the arg entity types
        # Examples: "chem_gene:increases^expression", "chem_disease:marker/mechanism"
        subj_type, obj_type = relation_label.split(":")[0].split("_")
        self.subj_type = self.STDD_ARG_TYPES[subj_type]
        self.obj_type = self.STDD_ARG_TYPES[obj_type]

        return

    def get_subj_entity(self) -> Tuple[str, str]:
        return self.subj_type, self.subj_eid

    def get_obj_entity(self) -> Tuple[str, str]:
        return self.obj_type, self.obj_eid

    def get_pretty_relation_label(self):
        r_flds = self.relation_label.split(":")[1].split("^")
        if len(r_flds) == 1:
            r_action, r_type = None, r_flds[0]
        else:
            r_action, r_type = r_flds[0], r_flds[1]

        if r_action:
            r_type += f" - {r_action}"

        return f"{self.subj_type}-{self.obj_type} : {r_type}"

    @classmethod
    def from_pubtator_line(cls, flds: List[str], from_ctd: bool = True) -> Tuple["BinaryRelationship", str]:
        """
        Relationship Fields in ChemDisGen PubTator format:
            DocID, Relation-Label, Subj-Entity-ID, Obj-Entity-ID
        """
        assert len(flds) == 4, \
            f"Incorrect number of fields for BinaryRelationship (expected 4, got {len(flds)}), flds = {flds}"

        docid, reln, s_id, o_id = flds

        return BinaryRelationship(s_id, o_id, reln, from_ctd=from_ctd), docid

    def write(self, docid, file: TextIO = sys.stdout):
        print(docid, self.relation_label, self.subj_eid, self.obj_eid,
              sep="\t", file=file)

    def __str__(self):
        return "BinaryRelationship(" + ", ".join([f"{fld} = {getattr(self, fld)}" for fld in self.__dict__]) + ")"
# /


class EntityMention:
    """
    Each mention here is a contiguous substring of the document text:
        mention-text = doc.get_text()[ch_start : ch_end]
    Typically, each mention is for one Entity, uniquely specified by (Entity-Type, Entity-ID).
    Some mentions may be composite -- mention more than one entity -- as in "ovarian and peritoneal cancer".

    Some entity recognition models (e.g. Pubtator Central) will on occasion recognize the type of a mention
    without resolving it to a particular entity ID. In such cases, the Entity-ID will be null (empty) or "-".

    NOTE: Does not handle multiple values in `entity_type`
    """

    def __init__(self, ch_start: int, ch_end: int, mention: str, entity_type: str, entity_ids: Optional[str],
                 mention_id: str = None, composite_mentions: Optional[str] = None):
        self.ch_start = ch_start
        self.ch_end = ch_end
        self.mention = mention
        self.entity_type = entity_type

        # Not usually seen in Pubtator format.
        # Direct supervision labeling will include this, e.g. DrugProt.
        self.mention_id = mention_id

        if not entity_ids or entity_ids == "-":
            self.entity_ids = None
        else:
            self.entity_ids = re.split(r"[;|]", entity_ids)

        # When more than one ConceptID is assigned to mention
        # e.g.  mention = "ovarian and peritoneal cancer"
        #       concept_id = D010051|D010534
        #       Additional field composite_mentions = "ovarian cancer|peritoneal cancer"
        # where: D010051 = Ovarian Neoplasms, D010534 = Peritoneal Neoplasms
        self.is_composite = self.entity_ids is not None and len(self.entity_ids) > 1

        self.composite_mentions = composite_mentions.split("|") if composite_mentions else None

        self._from_title = False
        return

    @property
    def is_from_title(self):
        return self._from_title

    @is_from_title.setter
    def is_from_title(self, from_title: bool):
        self._from_title = from_title
        return

    def is_unresolved_mention(self):
        """
        Whether this mention is not resolved (not linked to any Entity IDs)
        """
        return self.entity_ids is None

    def get_entity_ids(self):
        if not self.entity_ids:
            return ["-"]
        else:
            return self.entity_ids

    def get_entities(self) -> List[Tuple[str, str]]:
        return [(self.entity_type, eid) for eid in self.get_entity_ids()]

    def write(self, docid, file: TextIO = sys.stdout, composite_sep: str = "|"):
        """
        Output in PubTator format.
        Ignores `mention_id`.
        """
        print(docid, self.ch_start, self.ch_end, self.mention, self.entity_type,
              composite_sep.join(self.get_entity_ids()),
              sep="\t", end="", file=file)
        if self.composite_mentions:
            print("", "|".join(self.composite_mentions), sep="\t", end="", file=file)
        print(file=file)
        return

    def __str__(self):
        return "EntityMention(" + ", ".join([f"{fld} = {getattr(self, fld)}" for fld in self.__dict__]) + ")"

    @classmethod
    def from_pubtator_line(cls, flds: List[str]):
        """
        Relationship Fields in PubTator format:
            DocID, Char-Start-Index, Char-End-Index, Mention-text, Entity-Type, Entity-IDs [, Composite-Mentions ]
        where
            Entity-IDs = empty  ||  "-"  ||  Entity-ID [ SEP Entity-ID ]*
            Entity-ID = str
            SEP = "|"  ||  ";"
            Composite-Mentions = Mention [ "|" Mention ]*
            Mention = str

        Example of composite mentions:
            11752998	213	235	acute and chronic pain	Disease	D059787|D059350     acute pain|chronic pain

            Mention-text = acute and chronic pain
            Entity-IDs = D059787|D059350
            Composite-Mentions = acute pain|chronic pain
        """

        assert len(flds) in [5, 6, 7], \
            f"Incorrect number of fields for EntityMention (expected 5-7, got {len(flds)})"

        flds = [fld.strip() for fld in flds]

        docid, ch_start, ch_end, mention, ent_type = flds[:5]
        ent_id = None
        composite_mentions = None

        if len(flds) > 5:
            ent_id = flds[5]

        if len(flds) > 6 and flds[6]:
            composite_mentions = flds[6]

        ch_start = int(ch_start)
        ch_end = int(ch_end)

        ann = EntityMention(ch_start, ch_end, mention, ent_type, ent_id, composite_mentions)

        return ann, docid
# /


class AnnotatedDocument:
    def __init__(self, docid: str, title: Optional[str] = None, abstract: Optional[str] = None):
        self.docid = docid
        self.title: Optional[str] = title
        self.abstract: Optional[str] = abstract
        self.title_len = None
        self.mentions: List[EntityMention] = []
        self.relationships: List[BinaryRelationship] = []

        # Dict: (entity_type: str, entity_id: str) => List[EntityMention]
        self._entity_mentions_dict = defaultdict(list)

        self._is_sorted = True
        return

    def get_title_length(self):
        if self.title is None:
            return 0
        elif self.title_len is None:
            self.title_len = len(self.title)

        return self.title_len

    def get_text(self, sep="\n"):
        if self.title and self.abstract:
            return self.title + sep + self.abstract
        elif self.title:
            return self.title
        else:
            return self.abstract

    def add_entity_mention(self, ent_mention: EntityMention):
        self.mentions.append(ent_mention)
        for entity in ent_mention.get_entities():
            self._entity_mentions_dict[entity].append(ent_mention)

        self._is_sorted = False
        return

    def add_relationship(self, reln: BinaryRelationship, validate: bool = True):
        # Relationships are usually seen after all the entity mentions have been added
        if validate:
            assert reln.get_subj_entity() in self._entity_mentions_dict, \
                f"Subject entity {reln.get_subj_entity()} not found in doc {self.docid},\nfor relationship {reln}"
            assert reln.get_obj_entity() in self._entity_mentions_dict, \
                f"Object entity {reln.get_obj_entity()} not found in doc {self.docid},\nfor relationship {reln}"

        self.relationships.append(reln)
        return

    def add_annotation_pubtator(self, pbtr_line: str):
        flds = pbtr_line.strip().split("\t")
        if len(flds) > 2:
            if is_integral(flds[1]):
                mention, docid = EntityMention.from_pubtator_line(flds)
                assert docid == self.docid
                # By the time we see annotations, we have already seen the Title
                mention.is_from_title = mention.ch_end <= self.get_title_length()

                self.add_entity_mention(mention)
            else:
                reln, docid = BinaryRelationship.from_pubtator_line(flds)
                assert docid == self.docid
                self.add_relationship(reln)

        return

    def sort_mentions(self, force_resort: bool = False):
        if force_resort or not self._is_sorted:
            self.mentions.sort(key=lambda antn: (antn.ch_start, antn.ch_end))
            self._is_sorted = True
        return

    def get_title_mentions(self):
        title_len = self.get_title_length()
        return [men for men in self.mentions if men.ch_end <= title_len]

    def get_body_mentions(self):
        title_len = self.get_title_length()
        return [men for men in self.mentions if men.ch_start >= title_len]

    def get_mentioned_entities(self) -> Set[Tuple[str, str]]:
        return set(self._entity_mentions_dict.keys())

    def get_entity_mentions(self, entity_type: str, entity_id: str) -> Optional[List[EntityMention]]:
        """
        :return: List of EntityMention that mention the entity `(entity_type, entity_id)` in this doc,
            or None if no such mentions.
        """
        return self._entity_mentions_dict.get((entity_type, entity_id))

    def write(self, file: TextIO = sys.stdout, write_relationships: bool = False, composite_sep: str = "|"):
        """
        Output in Pubtator format.
        """
        if self.get_title_length() > 0:
            print(self.docid, "t", self.title, sep="|", file=file)
        if self.abstract:
            print(self.docid, "a", self.abstract, sep="|", file=file)

        for mention in self.mentions:
            mention.write(self.docid, file=file, composite_sep=composite_sep)

        if write_relationships:
            for reln in self.relationships:
                reln.write(self.docid, file=file)

        # Always end with an empty line
        print(file=file)
        return
# /


# -----------------------------------------------------------------------------
#   Functions
# -----------------------------------------------------------------------------

def is_integral(txt: str):
    return all(c in "1234567890" for c in txt)


def parse_pubtator_to_dict(pbtr_file: str, relns_file: str = None, encoding: str = 'UTF-8') \
        -> Dict[str, AnnotatedDocument]:
    return {doc.docid: doc
            for doc in parse_pubtator(pbtr_file, relns_file=relns_file, encoding=encoding)}


def parse_pubtator(pbtr_file: str, relns_file: str = None, encoding: str = 'UTF-8') -> List[AnnotatedDocument]:
    """
    Parse a PubTator file and extract AnnotatedDocument's.
    Restrict to docids if `docids_file` provided, which contains one DOCID per file.
    Annotations are sorted on position in text.
    """
    if pbtr_file.endswith(".gz"):
        with gzip.open(os.path.expanduser(pbtr_file)) as f:
            docs = parse_pubtator_opened_file(f, encoding=encoding)
    else:
        with open(os.path.expanduser(pbtr_file), encoding=encoding) as f:
            docs = parse_pubtator_opened_file(f)

    if relns_file:
        if relns_file.endswith(".gz"):
            with gzip.open(os.path.expanduser(relns_file)) as f:
                relationships = parse_relationships_opened_file(f, encoding=encoding)
        else:
            with open(os.path.expanduser(relns_file), encoding=encoding) as f:
                relationships = parse_relationships_opened_file(f)

        for doc in docs:
            for reln in relationships[doc.docid]:
                try:
                    doc.add_relationship(reln)
                except AssertionError as e:
                    print(e)
                    print("Skipping entry ...\n")

    return docs


def parse_pubtator_opened_file(f, encoding: str = 'UTF-8') -> List[AnnotatedDocument]:

    andocs = []
    curdoc = None
    lc = 0

    for line in f:
        lc += 1
        if isinstance(line, bytes):
            line = line.decode(encoding)

        line = line.strip()
        if line == '':
            if curdoc:
                curdoc.sort_mentions()
                andocs.append(curdoc)
                curdoc = None
            continue

        m = TITLE_ABSTR_PATT.match(line)
        if m:
            docid = m.group(1)

            if not curdoc:
                curdoc = AnnotatedDocument(docid)
            elif curdoc.docid != docid:
                raise ValueError("DocID mismatch at line {}".format(lc))

            text = line[m.end(0):]

            if m.group(2) == 't':
                curdoc.title = text
            else:
                curdoc.abstract = text
        else:
            try:
                curdoc.add_annotation_pubtator(line)
            except AssertionError as e:
                print("Error while processing PubTator file")
                print(e)
                print(f"Line {lc}:", line)
                print("Skipping entry ...\n")

    if curdoc:
        curdoc.sort_mentions()
        andocs.append(curdoc)

    return andocs


def parse_relationships_opened_file(f, encoding: str = 'UTF-8', from_ctd: bool = True) \
        -> Dict[str, List[BinaryRelationship]]:

    relationships = defaultdict(list)
    lc = 0
    for line in f:
        lc += 1
        if isinstance(line, bytes):
            line = line.decode(encoding)
        flds = line.strip().split("\t")
        try:
            reln, docid = BinaryRelationship.from_pubtator_line(flds, from_ctd=from_ctd)
            relationships[docid].append(reln)
        except Exception as e:
            print(f"Error at line {lc}: [{line.strip()}]")
            raise e

    return relationships


def parse_tsv_files_to_dict(abstracts_file: str, entities_file: str, relations_file: str) \
        -> Dict[str, AnnotatedDocument]:
    """
    Get data from separate TSV files, as in DrugProt.

    Fields in the TSV Files
    -----------------------

    abstracts_file:
        DocID, Title, Abstract

    entities_file:
        DocID, MentionID, EntityType, ch_start, ch_end, Mention-text

    relations_file:
        DocID, RelationType, "Arg1:{MentionID}",  "Arg2:{MentionID}"
    """
    docs_dict = dict()
    n_docs, n_mentions, n_relns = 0, 0, 0

    with open(abstracts_file) as f:
        for line in f:
            docid, title, abstract = line.strip().split("\t")
            docs_dict[docid] = AnnotatedDocument(docid, title=title.strip(), abstract=abstract.strip())
            n_docs += 1

    with open(entities_file) as f:
        for line in f:
            docid, mention_id, entity_type, ch_start, ch_end, mention = line.strip().split("\t")
            ch_start = int(ch_start)
            ch_end = int(ch_end)
            ent_men = EntityMention(ch_start, ch_end, mention, entity_type, entity_ids=None, mention_id=mention_id)
            docs_dict[docid].add_entity_mention(ent_men)
            n_mentions += 1

    with open(relations_file) as f:
        for lc, line in enumerate(f, start=1):
            docid, reln_type, arg1, arg2 = line.strip().split("\t")

            assert arg1.startswith("Arg1") and arg2.startswith("Arg2"), \
                f"Bad line format at line {lc}: bad args"

            # Strip the "Arg*" prefixes
            reln = BinaryRelationship(arg1[5:], arg2[5:], reln_type)
            docs_dict[docid].add_relationship(reln)
            n_relns += 1

    print()
    print(f"Nbr docs read      = {n_docs:6,d}")
    print(f"Nbr mentions read  = {n_mentions:6,d}")
    print(f"Nbr relations read = {n_relns:6,d}")

    return docs_dict
