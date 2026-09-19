"""Copy the audited official XLSX inputs locally; match by SHA-256, never by name alone."""
from pathlib import Path
import argparse, hashlib, json, shutil

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('official_folder',type=Path,help='Folder containing extracted official C-question attachments')
    args=parser.parse_args()
    source=args.official_folder.resolve()
    if not source.is_dir(): parser.error('Input folder does not exist')
    candidates={}
    for path in source.rglob('*.xlsx'):
        candidates.setdefault(hashlib.sha256(path.read_bytes()).hexdigest(),path)
    manifest=json.loads((ROOT/'INPUTS.json').read_text('utf-8'))['files']
    missing=[name for name,digest in manifest.items() if digest not in candidates]
    if missing: raise SystemExit('No exact hash match for: '+', '.join(missing))
    for name,digest in manifest.items():
        target=ROOT/name; target.parent.mkdir(parents=True,exist_ok=True)
        if candidates[digest].resolve()!=target.resolve(): shutil.copyfile(candidates[digest],target)
    print(f'Imported {len(manifest)} hash-verified files into ignored local input directories.')

if __name__=='__main__':main()
