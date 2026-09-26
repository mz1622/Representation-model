"""JSON-file interface for research predict / encode / retrieve_names."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_r0 import write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--request",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    request=json.loads(args.request.read_text(encoding="utf-8-sig"))
    model=NutritionModel(args.checkpoint)
    action=request.pop("action")
    allowed={"predict","encode","retrieve_names"}
    if action not in allowed:raise ValueError(f"Action must be one of {allowed}")
    result=getattr(model,action)(**request)
    if action=="encode":result=result.tolist()
    write_json(args.output,{"action":action,"result":result,"status":"research-only; labels and external generalization not confirmed"})

if __name__=="__main__":main()
