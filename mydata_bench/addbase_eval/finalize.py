"""Collect the full matrix once the monitored scheduler ends successfully."""
import json
import subprocess
import sys
import time
from .prepare import OUT,ROOT


def main():
    marker=OUT/'research/scheduler_completion.json'
    while not marker.exists():time.sleep(30)
    state=json.loads(marker.read_text())
    bad={name:r for name,r in state['finished'].items() if not r.get('complete')}
    if bad:raise RuntimeError(f'Experiments require repair before scoring: {bad}')
    for module in ['audit','score','overlap','write_records']:
        command=[sys.executable,'-u','-m',f'mydata_bench.addbase_eval.{module}']
        print('Running',command,flush=True)
        subprocess.run(command,cwd=ROOT,check=True)
    print('All quantitative artifacts ready for final interpretation.',flush=True)


if __name__=='__main__':main()
