import argparse 
import json 
import os 
import numpy as np 
from typing import List, Dict, Any
from pathlib import Path 

parser = argparse.ArgumentParser()
parser.add_argument("--model_save_name", type=str, required=True)
parser.add_argument(
    "--meta_result_save_dir", default=None, type=str
)
args = parser.parse_args()

meta_result_dir = Path(args.meta_result_save_dir) / args.model_save_name / "RMB"
pair_wise_harmless = meta_result_dir / "Pairwise_set_Harmlessness" / "score_result" / "Final_score.json"
pair_wise_helpfulness = meta_result_dir / "Pairwise_set_Helpfulness" / "score_result" / "Final_score.json"

bon_harmless = meta_result_dir / "BoN_set_Harmlessness" / "score_result" / "Final_score.json"
bon_helpfulness = meta_result_dir / "BoN_set_Helpfulness"  / "score_result" / "Final_score.json"

with open(pair_wise_harmless) as json_file:
    pair_wise_harmless_doc = json.load(json_file)
with open(pair_wise_helpfulness) as json_file:
    pair_wise_helpful_doc = json.load(json_file)
with open(bon_harmless) as json_file:
    bon_harmless_doc = json.load(json_file)
with open(bon_helpfulness) as json_file:
    bon_helpful_doc = json.load(json_file)

# print(pair_wise_helpful_doc)
Final_result = {
    "Pairwise Helpfulness": pair_wise_helpful_doc[0]["Overall_accuracy"],
    "Pairwise Harmlessness": pair_wise_harmless_doc[0]["Overall_accuracy"],
    "BoN Helpfulness": bon_helpful_doc[0]["BoN Final Accuracy"],
    "BoN Harmlessness":bon_harmless_doc[0]["BoN Final Accuracy"]
}

Final_result["Accuracy Averaged from Four Sections"] = np.mean(list(Final_result.values()))

if args.meta_result_save_dir is not None:
    Meta_result_save_dir = Path(args.meta_result_save_dir) / args.model_save_name / "RMB"
    Meta_result_save_dir.mkdir(exist_ok=True, parents=True)

    with open(Meta_result_save_dir / "META_RESULT.json", 'w') as f:
        json.dump(Final_result, f, indent=4)