Contents of ChemDisGene/data/curated

abstracts.txt.gz
	Curated abstracts with entity mentions, in PubTator format.

approved_relns_ctd_v?.tsv.gz
	CTD-derived relationships approved by curation process.
	"_v1" = version after review of singleton relationships; else "_v0".

approved_relns_new_v?.tsv.gz
	New relationships approved by curation process.
	"_v1" = version after review of singleton relationships; else "_v0".

drugprot_pmids.txt.gz
	List of PubMedID's of documents that are also in DrugProt. One per line.

curated_stats.txt
	Basic stats on Curated data.
	To replicate, run from the Python dir:
	[Python] $> python -m chemdisgene.analysis.datastats curated_basic ../data/curated

