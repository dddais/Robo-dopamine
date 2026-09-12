"""Download public primary sources; never send local research data."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup

OUT = Path('results/mydata_bench/experiments_v2_corssmodel/auto_research/session_20260911/references')
IDS = ['2311.02262','2407.21771','2506.14766','2605.04641','2606.14703',
       '2607.17994','2604.09364','2608.17095','2608.03711','2503.06287',
       '2606.09749','1905.09418','2311.16922','2603.02115','2603.28730']

def fetch(aid):
    result = {'arxiv_id':aid, 'sources':[]}
    for kind in ['abs','html']:
        url = f'https://arxiv.org/{kind}/{aid}'
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            content = response.text
            path = OUT / f'{aid}_{kind}.html'
            with path.open('x') as f: f.write(content)
            soup = BeautifulSoup(content, 'html.parser')
            for tag in soup(['script','style','nav']): tag.decompose()
            plain = soup.get_text(' ', strip=True)
            with (OUT/f'{aid}_{kind}.txt').open('x') as f: f.write(plain)
            result['sources'].append({'url':response.url, 'sha256':hashlib.sha256(content.encode()).hexdigest(),
                                     'bytes':len(content),'title':soup.title.get_text() if soup.title else None})
            if kind == 'abs':
                abstract=soup.find('blockquote',class_='abstract')
                result['abstract']=abstract.get_text(' ',strip=True) if abstract else None
        except Exception as exc:
            result['sources'].append({'url':url,'error':str(exc)})
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result

if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=5) as pool: rows=list(pool.map(fetch,IDS))
    with (OUT/'source_manifest.json').open('x') as f: json.dump(rows,f,ensure_ascii=False,indent=2)
