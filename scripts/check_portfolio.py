"""Check the distributable artifact, not a replacement for numerical reproduction."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import hashlib,json,re,subprocess
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    manifest=json.loads((ROOT/'publication-manifest.json').read_text('utf-8'))
    paths=[r['path'] for r in manifest['files']]
    assert len(paths)==len(set(paths)), 'Duplicate manifest entries'
    issues=[]
    links=0
    for row in manifest['files']:
        p=ROOT/row['path']
        if not p.is_file() or sha(p)!=row['sha256']:
            issues.append(f'Missing or modified publication file: {row["path"]}')
            continue
        if p.suffix.lower() in {'.xlsx','.xls','.zip','.pyc','.env'} or any(x in p.relative_to(ROOT).parts for x in ['.venv','.local','.reproduction','__pycache__']):
            issues.append(f'Forbidden publication path: {row["path"]}')
        if p.suffix.lower() in {'.md','.json','.py','.txt','.csv'}:
            content=p.read_text('utf-8-sig')
            secret_patterns=[r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'sk-[A-Za-z0-9_-]{25,}',r'C:[\\/]+Users[\\/]+[^\s]+']
            if any(re.search(pattern,content) for pattern in secret_patterns):
                issues.append(f'Potential credential or private absolute path: {row["path"]}')
        if p.suffix.lower()=='.md':
            content=re.sub(r'```.*?```','',content,flags=re.S)
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',content):
                target=target.strip().strip('<>')
                u=urlsplit(target)
                if u.scheme or u.netloc or not u.path:continue
                dest=(p.parent/unquote(u.path)).resolve()
                if not dest.is_relative_to(ROOT) or not dest.exists():
                    issues.append(f'Broken local link in {row["path"]}: {target}')
                links+=1
    for row in json.loads((ROOT/'assets/manifest.json').read_text('utf-8'))['inputs']+json.loads((ROOT/'assets/manifest.json').read_text('utf-8'))['outputs']:
        assert sha(ROOT/row['path'])==row['sha256'],f'Chart artifact drift: {row["path"]}'
    micro=json.loads((ROOT/'projects/microgrid-2026/provenance.json').read_text('utf-8'))
    for row in micro['files']:
        if row.get('public',True):assert sha(ROOT/row['destination'])==row['sha256']
    base=ROOT/'projects/agriculture-2024'
    agri=json.loads((base/'provenance.json').read_text('utf-8'))
    for row in agri['records']:
        assert sha(base/row['public_relative_path']).lower()==row['public_sha256'].lower(),row['public_relative_path']
    application=json.loads((ROOT/'application/experience.json').read_text('utf-8'))
    md=(ROOT/'application/README.md').read_text('utf-8')
    counts={}
    for item in application['projects']:
        assert item['title'] in md and item['role']=='建模方案审查、研究推进与成果整理'
        counts[item['id']]={}
        for field,bounds in [('short_text',(150,250)),('long_text',(400,700))]:
            text=item[field]
            count=len(text.replace('\n',''))
            assert bounds[0]<=count<=bounds[1],(item['id'],field,count)
            assert text in md,f'Application copy mismatch: {item["id"]}/{field}'
            assert count<1000
            counts[item['id']][field]=count
    pdf=PdfReader(ROOT/'output/pdf/modeling-analytics-portfolio.pdf')
    assert len(pdf.pages)==6
    assert all(len(p.extract_text() or '')>300 for p in pdf.pages)
    assert not pdf.attachments
    assert json.loads((ROOT/'projects/microgrid-2026/verification/representative.json').read_text('utf-8'))['status']=='PASS'
    assert json.loads((base/'evidence/verification.json').read_text('utf-8'))['status']=='PASS'
    assert not issues,'\n'.join(issues)
    print(json.dumps({'status':'PASS','public_files':len(paths)+1,'local_links_checked':links,'pdf_pages':6,'application_characters':counts,'scope':'Hashes, links, manifest, text consistency and PDF structure; see docs/verification.md for executed numerical checks.'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
