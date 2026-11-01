import argparse
from pathlib import Path
import json
from nisep.data import generate_dataset,load_data
from nisep.train import train_sequences,train_cnn,train_recovery
from nisep.evaluate import evaluate_all
p=argparse.ArgumentParser(); p.add_argument("--missions",type=int,default=1200); p.add_argument("--epochs",type=int,default=8); p.add_argument("--seed",type=int,default=181)
a=p.parse_args()
generate_dataset("data",a.missions,24,a.seed)
t,s,_=load_data("data"); Path("artifacts").mkdir(exist_ok=True)
r={"sequence_models":train_sequences(t,"artifacts",a.epochs),
   "cnn":train_cnn("artifacts",max(3,a.epochs//2)),
   "recovery":train_recovery(s,"artifacts",a.epochs)}
r["evaluation"]=evaluate_all(t,s,"artifacts")
Path("artifacts/training_summary.json").write_text(json.dumps(r,indent=2))
print("NISEP-181 pipeline completed.")
print(json.dumps(r,indent=2))
