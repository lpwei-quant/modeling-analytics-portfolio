"""Prepare an ignored, isolated copy of the historical calculation workspace."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys

BASE=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attachment-1',type=Path,required=True)
    parser.add_argument('--attachment-2',type=Path,required=True)
    args=parser.parse_args()
    expected=['b799a137294a5c5497fed9667c5dfed6c967f1fb180316c5adc8509cfa6f0932','869081a3ab47d3bf8d0955106b622aaf0fd2c068fada7948da69b20ebf1d00ce']
    for path,digest in zip([args.attachment_1,args.attachment_2],expected):
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise SystemExit('Official input SHA-256 mismatch')
    dest=BASE/'.reproduction/full-model'
    if dest.exists():raise SystemExit('Workspace already exists; preserve it rather than overwrite historical outputs.')
    for folder in ['src','scripts','data','research','results']:(dest/folder).mkdir(parents=True,exist_ok=True)
    for source in (BASE/'src').glob('*.py'):shutil.copyfile(source,dest/'src'/source.name)
    for name in ['run_q1.py','run_q2.py','run_q3.py','run_q2_deadline_closure.py','reproduce_core.py']:
        shutil.copyfile(BASE/'scripts'/name,dest/'scripts'/name)
    for source in (BASE/'historical-inputs').iterdir():shutil.copyfile(source,dest/'research'/source.name)
    shutil.copyfile(args.attachment_1,dest/'data/附件1 (1).xlsx')
    shutil.copyfile(args.attachment_2,dest/'data/附件2 (2).xlsx')
    for name in ['q2_final_plan.csv','q3_final_plan.csv']:shutil.copyfile(BASE/'data'/name,dest/'results'/name)
    sys.path.insert(0,str(dest/'src'))
    from q1_data import load_q1_data
    data=load_q1_data(dest)
    report={'status':'PREPARED','plots':len(data.plots),'crops':len(data.crops),'historical_optimizations_rerun':False,'historical_q2_q3_freeze_records_included':False}
    (dest/'preparation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
