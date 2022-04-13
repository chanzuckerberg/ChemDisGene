import json
import sys
import os
import gzip


data_dir = sys.argv[1] # "../data/"
output_dir = sys.argv[2] # "./data/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def process(abstract_file, relation_files, output_file):
    relations = {}
    count = 0
    for relation_file in relation_files:
        for line in gzip.open(relation_file, 'rb'):
            l = (line).decode('ascii').strip("\n").split("\t")
            if len(l) == 4:
                docid, rel, sid, oid = l
                if docid not in relations:
                    relations[docid] = []
                relations[docid].append({"type": rel, "subj": sid, "obj": oid})
                count += 1
        print("number of relations", count)

    corpus = gzip.open(abstract_file, 'rb').read()
    corpus = (corpus).decode('ascii').split("\n\n")
    data = []
    for doc in corpus:
        lines = doc.strip("\n").split("\n")
        if lines == ['']:
            continue
        docid = lines[0].split("|")[0]
        title = lines[0].split("|t|")[1] if len(lines[0].split("|t|")) == 2 else ""
        abstract = lines[1].split("|a|")[1] if len(lines[1].split("|a|")) == 2 else ""
        
        entities = []
        for ent in lines[2:]:
            _, start, end, mention, etype, eid = ent.split("\t")
            entities.append({"start": int(start), "end": int(end), "mention": mention, "type": etype, "id": eid})
        data.append({"docid": docid, "title": title, "abstract": abstract, "entity": entities})
        if docid in relations:
            data[-1]["relation"] = relations[docid]
        else:
            data[-1]["relation"] = []

    print("number of documents", len(data))

    open(output_file,"w").write(json.dumps(data, indent="\t"))


def relation_map(train_file, output_file):

    data = json.loads(open(train_file).read())
    rel_count = {}
    for d in data:
        for r in d["relation"]:
            if r["type"] not in rel_count:
                rel_count[r["type"]] = 0
            rel_count[r["type"]] += 1

    rel_count = sorted(list(rel_count.items()), key=lambda x:x[1], reverse=True)
    rel_dict = {}
    for r, c in rel_count:
        if r not in rel_dict:
            rel_dict[r] = len(rel_dict)

    open(output_file, "w").write(json.dumps(rel_dict, indent="\t"))


process(data_dir+"/ctd_derived/train_abstracts.txt.gz", [data_dir+"/ctd_derived/train_relationships.tsv.gz"], output_dir+"/train.json")
process(data_dir+"/ctd_derived/dev_abstracts.txt.gz", [data_dir+"/ctd_derived/dev_relationships.tsv.gz"], output_dir+"/valid.json")
process(data_dir+"/ctd_derived/test_abstracts.txt.gz", [data_dir+"/ctd_derived/test_relationships.tsv.gz"], output_dir+"/test.json")
process(data_dir+"/curated/abstracts.txt.gz", [data_dir+"/curated/approved_relns_ctd_v1.tsv.gz"], output_dir+"/test.anno_ctd.json")
process(data_dir+"/curated/abstracts.txt.gz", [data_dir+"/curated/approved_relns_ctd_v1.tsv.gz", data_dir+"/curated/approved_relns_new_v1.tsv.gz"], output_dir+"/test.anno_all.json")
relation_map(output_dir+"/train.json", output_dir+"/relation_map.json")
