# ChemDisGene/Python

Python code for reading the ChemDisGene corpora.

## Requirements

This code was tested using Python ver. 3.8.


## Reading the corpora

The API for reading the data is in [chemdisgene/data/pubtator.py](chemdisgene/data/pubtator.py). The example below shows the following steps:

1. Read the test split of the CTD-derived corpus, and return a dictionary of `AnnotatedDocument` indexed on document-id. 
2. Get the `AnnotatedDocument` for Pubmed ID `30074247`.
3. Get the list of mentions (`EntityMention` objects) for the entity with type `Chemical` and ID `MESH:C580853`.

```python
from chemdisgene.data.pubtator import parse_pubtator_to_dict

# Step 1
data_dir = "../data/ctd_derived"
abstracts_file = f"{data_dir}/test_abstracts.txt.gz" 
relns_file = f"{data_dir}/test_relationships.tsv.gz"

docs_dict = parse_pubtator_to_dict(abstracts_file, relns_file)

# Step 2
pmid = "30074247"
doc = docs_dict[pmid]

# Step 3
ent_type, ent_id = "Chemical", "MESH:C580853"
mentions = doc.get_entity_mentions(ent_type, ent_id)
```

The next example steps through each relationship in a document, retrieves all the mentions of the argument entities, and processes them.

```python
for reln in doc.relationships:
    subj_mentions = doc.get_entity_mentions(reln.subj_type, reln.subj_eid)
    obj_mentions = doc.get_entity_mentions(reln.obj_type, reln.obj_eid)
    process_reln(reln, subj_mentions, obj_mentions)
```

For more examples, take a look at [chemdisgene/analysis/datastats.py](chemdisgene/analysis/datastats.py).