"""Audit actual name tokenizer truncation without loading or evaluating test rows."""
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,write_json,digest
from foodcomp.research_text import REVISION,prepare_names

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    data=ResearchData(ROOT/"data/processed"/VERSION)
    _,cache=prepare_names(data,ROOT)
    path=ROOT/"data/cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots"/REVISION
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    names=sorted(set(data.profiles.original_name.fillna("").astype(str)))
    lengths=np.array([len(x) for x in tokenizer(names,truncation=False,add_special_tokens=True)["input_ids"]])
    pd.DataFrame({"name":names,"token_length":lengths}).to_csv(args.output_dir/"name_lengths.csv",index=False)
    write_json(args.output_dir/"summary.json",{"unique_names":len(names),"truncated_at_128":int((lengths>128).sum()),
        "max_tokens":int(lengths.max()),"median_tokens":float(np.median(lengths)),"empty_names":sum(not n.strip() for n in names),
        "cache_manifest_sha256":digest(cache/"manifest.json"),"data_manifest_sha256":digest(data.root/"manifest.json"),
        "text_field":"original_name only","cache_data_fingerprint":"profile IDs, exact name contents and training row indices; numerical data version independently pinned by checkpoint data hash",
        "complete_test_opened":False})

if __name__=="__main__":main()
