Contents of ChemDisGene/data/curated

abstracts.txt.gz
	Curated abstracts with entity mentions, in PubTator format.

approved_relns_ctd_v0.tsv.gz
	CTD-derived relationships approved by curation process.

approved_relns_new_v0.tsv.gz
	New relationships approved by curation process.

drugprot_pmids.txt.gz
	List of PubMedID's of documents that are also in DrugProt. One per line.

curated_stats.txt
	Basic stats on Curated data.
	To replicate, run from the Python dir:
	[Python] $> python -m chemdisgene.analysis.datastats curated_basic ../data/curated

