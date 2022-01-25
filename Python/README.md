# ChemDisGene/Python

Python code for reading the ChemDisGene corpora.

## Requirements

This code was tested using Python ver. 3.8.


## Reading the corpora

The API for reading the data is in `chemdisgene/data/pubtator.py`. Here is an example which reads the test split of the CTD-derived corpus, and returns a dictionary of `AnnotatedDocument` indexed on document-id:

```python
from chemdisgene.data.pubtator import parse_pubtator_to_dict
data_dir = "../data/ctd_derived"
abstracts_file = ".{data_dir}/test_abstracts.txt.gz" 
relns_file = "{data_dir}/test_relationships.tsv.gz"
docs_dict = parse_pubtator_to_dict(abstracts_file, relns_file)
```

For more examples, take a look at `chemdisgene/analysis/datastats.py`.