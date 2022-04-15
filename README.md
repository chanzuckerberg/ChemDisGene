# ChemDisGene: A Biomedical Relation Extraction Dataset

This is a public release of the _ChemDisGene_ dataset, a collection of Biomedical research abstracts annotated with mentions of Chemical, Disease and Gene/Gene-product entities, and pairwise relationships between those entities. [CZI Science](https://chanzuckerberg.com/science/) 
is releasing this data to promote NLP research on Relation Extraction from Biomedical text.


## Project Status

This project is stable and maintained, but not actively under development.


## Introduction

The _ChemDisGene_ dataset contains two corpora:

* [CTD-derived](data/ctd_derived/): A corpus of ~80k abstracts, with entity mentions from PubTator Central, and automatically aligned noisy relationship labels derived from CTD.
* [Curated](data/curated/): A corpus of 523 abstracts, with entity mentions from PubTator Central, and relationship labels manually curated by a team of biologists.

Details on how this corpus was derived from CTD and further curated are in the accompanying paper (see below).

Annotation guidelines developed for the curation task are included [here](curation/AnnotationGuidelines.pdf).


### Directory Structure

* [data/](data)
	* [ctd_derived/](data/ctd_derived/): CTD-derived corpus.
	* [curated/](data/curated/): Curated corpus.
* [curation/](curation/): Contains annotation guidelines.
* [Python/](Python/): Python code for reading the data.
* [baseline/](baseline/): Code for the baseline models in our paper.


### Cloning the Repository

The data files are stored using `git lfs`. Follow [these](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage) instructions to install `git lfs` on your machine. Then issue the following command:

```
$ git lfs clone https://github.com/chanzuckerberg/ChemDisGene.git
```

This will create a new directory called `ChemDisGene` and clone the repository into that directory.


## Entities and Relationships

**Entity mentions** are linked to the following public ontologies:

* Chemicals are linked to [MeSH&reg;](https://www.nlm.nih.gov/mesh/meshhome.html)
* Diseases are linked to [MeSH&reg;](https://www.nlm.nih.gov/mesh/meshhome.html) and [OMIM&reg;](https://www.omim.org).
* Genes/Gene-products are linked to [NCBI Gene](https://www.ncbi.nlm.nih.gov/gene).

The dataset uses a total of 18 **relation types**, based on the classes described below (definitions are from [CTD Glossary](http://ctdbase.org/help/glossary.jsp)). Some of these classes are further qualified by a _degree_.

* **Chemical-Disease**:
	* *marker/mechanism*: A chemical that correlates with a disease.
	* *therapeutic*: A chemical that has a known or
potential therapeutic role in a disease.

* **Chemical-Gene**: Each qualified by a degree.
	* *activity*: An elemental function of a molecule. Degrees: increases, decreases, or affects when the direction is not indicated.
	* *binding*: A molecular interaction. Degrees: affects.
	* *expression*: Expression of a gene product. Degrees: increases, decreases, affects.
	* *localization*: Part of the cell where a molecule resides. Degrees: affects.
	* *metabolic processing*: The biochemical alteration of a molecule’s structure (not including changes in expression, stability, folding, localization, splicing, or transport). Degrees: increases, decreases, affects.
	* *transport*: The movement of a molecule into or out of a cell. Degrees: increases, decreases, affects.

* **Gene-Disease**:
	* *marker/mechanism*: A gene that may be a biomarker of a disease or play a role in the etiology of a disease.
	* *therapeutic*: A gene that is or may be a ther- apeutic target in the treatment a disease.


## ChemDisGene Data Format

The [CTD-derived corpus](data/ctd_derived/) is derived from the February 2021 dump of CTD, and is partitioned into *train*, *dev* and *test* subsets based on paper publication year: 2018 for *dev* and the years 2019, 2020 for *test*. Each split consists of an abstracts file containing the text of the abstracts and all the entity mentions, and a relationships file containing all the identified relationships.

The [Curated corpus](data/curated/) in *ChemDisGene* contains 271 documents from the *test* split of the CTD-derived corpus, and another set of older 252 abstracts taken from [DrugProt](https://biocreative.bioinformatics.udel.edu/tasks/biocreative-vii/track-1/) that had also been annotated by CTD. These 252 documents are not in the CTD-derived corpus.

The [Curated corpus](data/curated/) consists of the following files:

* abstracts with entity mentions
* approved relationships derived from CTD
* additional approved relationships
* file containing PubMed&reg; IDs of the DrugProt abstracts

All 'abstract' files are in PubTator format. All relationship files also follow the PubTator format for relationships. Both are described below.

### The PubTator format for Abstracts and Entity Mentions

Abstracts are annotated in [PubTator](http://bioportal.bioontology.org/ontologies/EDAM?p=classes&conceptid=format_3783)
format. Each document ends with a blank line, and is represented as (without the spaces):

```
PMID | t | Title text
PMID | a | Abstract text
PMID TAB StartIndex TAB EndIndex TAB MentionTextSegment TAB EntityType TAB EntityID
...
```

The first two lines present the Title and Abstract texts (no line-breaks or tabs in the _text_). 
Subsequent lines present the mentions, one per line.
The _StartIndex_ and _EndIndex_ are 0-based character indices into the document text, constructed
by concatenating the Title and Abstract, separated by a SPACE character. The _MentionTextSegment_
is the actual mention between those character positions. The _EntityID_ is the unique identifier for that entity in the corresponding ontology, and the _EntityType_ is one of "Chemical", "Disease" or "Gene". If the entity is linked to more than one ID, then that field 
contains a |-separated list of all the IDs. Occasionally, PubTator Central identifies the type of a mention but is unable to link it. This is indicated by the special ID "-".

Here is an example:

```
10226872|t|Disodium cromoglycate does not prevent terbutaline-induced desensitization of beta 2-adrenoceptor-mediated cardiovascular in vivo functions in human volunteers.
10226872|a|In humans, prolonged administration of the beta 2-adrenoceptor agonist terbutaline leads to a desensitization of beta 2-adrenoceptor-mediated cardiovascular responses, which can be blunted by concomitant administration of the antianaphylactic drug ketotifen. This study investigated the effect of disodium cromoglycate, another antiallergic drug, on terbutaline-induced desensitization of beta-adrenoceptor-mediated cardiovascular and noncardiovascular responses. In a double-blind, placebo-controlled, randomized design, nine healthy male volunteers received disodium cromoglycate (4 x 200 mg/day, p.o.) or placebo for 3 weeks with terbutaline (3 x 5 mg/day, p.o.) administered concomitantly during the last 2 weeks. beta 2-Adrenoceptor cardiovascular function was assessed by the increase in heart rate and reduction of diastolic blood pressure induced by an incremental intravenous infusion of the unselective beta-adrenoceptor agonist isoprenaline; beta 1-adrenoceptor cardiovascular function was assessed by exercise-induced tachycardia. Tremulousness was monitored as a beta 2-adrenoceptor-mediated noncardiovascular effect. After 2 weeks' administration of terbutaline, there was a marked and significant (p &lt; 0.001) attenuation of isoprenaline-induced tachycardia (mean percentage attenuation, 53.3%) and of the isoprenaline-induced decrease in diastolic blood pressure (mean percentage attenuation, 55.6%). Exercise-induced tachycardia also was significantly (p &lt; 0.001) blunted, but the magnitude of this attenuation was only very small (mean attenuation, 5.6%). Disodium cromoglycate affected neither the rightward shift of beta 2-adrenoceptor-mediated responses nor the small rightward shift in beta 1-adrenoceptor-mediated exercise tachycardia after 2 weeks' administration of terbutaline. Tremulousness observed during the first few days of terbutaline administration disappeared after 4 to 8 days, indicating development of desensitization of beta 2-adrenoceptor-mediated noncardiovascular responses. This was not prevented by disodium cromoglycate. These results confirm that long-term beta 2-adrenoceptor agonist therapy leads to a desensitization of beta 2-adrenoceptor-mediated cardiovascular and noncardiovascular effects in humans in vivo. However, unlike ketotifen, cromolyn sodium is not able to attenuate this desensitization.
10226872        0       21      Disodium cromoglycate   Chemical        MESH:D004205
10226872        39      50      terbutaline     Chemical        MESH:D013726
10226872        78      97      beta 2-adrenoceptor     Gene    154
10226872        204     223     beta 2-adrenoceptor     Gene    154
...
```

In this example, the Title is 160 characters long. The first mention occurs in the title, and is for the Chemical
"Disodium cromoglycate" whose MeSH&reg; id is _MESH:D004205_. The fourth mention in the list is from the body of the abstract, and is for the Gene "beta 2-adrenoceptor" linked to the NCBI Gene id "154". There is a previous mention of the same gene, in the title.


### Format for Relationships

Each line in a relationships file indicates one relationship in one of the corresponding documents. This is a TAB-separated file, with the following columns:

* PubMedID: PubMed&reg; ID for the document in which this relationship occurs.
* Relation-Type: One of 18 relation type labels
* Subject-Entity-ID: Unique ontology identifier for the subject entity
* Object-Entity-ID: Unique ontology identifier for the subject entity

Relationships are associated with the document, but not specific entity mentions (distant labeling). Here is an example showing three relationships associated with the abstract with PubMed&reg; ID "10226872":

```
10226872        chem_disease:marker/mechanism   MESH:D007545    MESH:D013610
10226872        chem_gene:affects^binding       MESH:D013726    154
10226872        chem_gene:increases^activity    MESH:D013726    154
```

The first relationship in the example is for the relation type "_Chemical-Disease: marker/mechanism_", with argument entities "MESH:D007545" (a Chemical) and "MESH:D013610" (a Disease). The relation type in the second relationship is "_Chemical-Gene: affects-binding_", where "_affects_" is the degree of "_binding_".


### Looking up Entities in their Ontologies

The following URL formats can be used to look up details about each entity (`{entid}` is the identifier without the ontology name):

* MeSH&reg;: `https://meshb.nlm.nih.gov/record/ui?ui={entid}`
* OMIM&reg;: `https://www.omim.org/entry/{entid}`
* NCBI Gene: `https://www.ncbi.nlm.nih.gov/gene/{entid}`

For example, here is the link for the Chemical "Isoproterenol": [MESH:D007545](https://meshb.nlm.nih.gov/record/ui?ui=D007545).


## How to Cite

If you use _ChemDisGene_, please cite the following paper:

Dongxu Zhang, Sunil Mohan, Michaela Torkar and Andrew McCallum.
*A Distant Supervision Corpus for Extracting Biomedical Relationships Between Chemicals, Diseases and Genes*.
In LREC 2022. [[arXiv](https://arxiv.org/abs/2201.11903)]

```
@InProceedings{zhang-etal:2022:LREC,
  author = {Dongxu Zhang and Sunil Mohan and Michaela Torkar and Andrew McCallum},
  title = {A Distant Supervision Corpus for Extracting Biomedical Relationships Between Chemicals, Diseases and Genes},
  booktitle = {Proceedings of The 13th Language Resources and Evaluation Conference},
  month = {June},
  year = {2022},
  address = {Marseille, France},
  publisher = {European Language Resources Association},
}
```

## License

This data is being released under the [CC0 license](https://creativecommons.org/publicdomain/zero/1.0/).

The abstracts in the dataset were selected from those available from [PubMed&reg; / Medline&reg;](https://www.nlm.nih.gov/databases/download/pubmed_medline.html) during February 2021.
Users are referred to that source for the most current and accurate version of the text for the corresponding papers.

Entity mentions were obtained using [PubTator Central](https://www.ncbi.nlm.nih.gov/research/pubtator/), and the relationships are based on the [Comparative Toxicogenomics Database](http://ctdbase.org).

Curated Chemical–Gene, Chemical–Disease and Gene–Disease interactions data were retrieved from the [Comparative Toxicogenomics Database](http://ctdbase.org/) (CTD), MDI Biological Laboratory, Salisbury Cove, Maine, and NC State University, Raleigh, North Carolina. [February, 2021].


## Feedback, Questions

If you have any comments, questions or issues, please post a note in 
[GitHub issues](https://github.com/chanzuckerberg/ChemDisGene/issues).

## Security Issues

[Reporting security issues](SECURITY.md)
