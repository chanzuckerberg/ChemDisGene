Contents of ChemDisGene/data/ctd_derived/

dev_abstracts.txt.gz
	Abstracts with entity mentions in Pubtator format for the 'dev' split.

dev_relationships.tsv.gz
	Relationships in Pubtator format for the 'dev' split.

test_abstracts.txt.gz
	Abstracts with entity mentions in Pubtator format for the 'test' split.

test_relationships.tsv.gz
	Relationships in Pubtator format for the 'test' split.

train_abstracts.txt.gz
	Abstracts with entity mentions in Pubtator format for the 'training' split.

train_relationships.tsv.gz
	Relationships in Pubtator format for the 'training' split.

ctd_stats.txt
	Basic stats on CTD-derived data.
	To replicate, run from the Python dir:
	[Python] $> python -m chemdisgene.analysis.datastats ctd_basic ../data/ctd_derived

