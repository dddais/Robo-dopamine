"""Read-only progress and latest-batch error inspection for long matrix runs."""
from collections import Counter, deque
import json
from pathlib import Path
import re


OUT = Path(__file__).resolve().parents[2] / 'results/mydata_bench/experiments_v2_addbase'


def verify_process(pid, module):
    """Read the live process identity; a saved scheduler flag is not sufficient."""
    try:
        proc = Path('/proc') / str(pid)
        state = (proc / 'stat').read_text().split()[2]
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        return state != 'Z' and module in command
    except (FileNotFoundError, ProcessLookupError):
        return False


def main():
    research = OUT / 'research'
    state = json.loads((research / 'scheduler_status.jsonl').read_text().splitlines()[-1])
    running_outputs = {'sole_official' if '_helper_' in job['name'] else job['name']
                       for job in state['running'].values()}
    completed = sum(v.get('complete', False) for k, v in state['finished'].items()
                    if '_helper_' not in k and k not in running_outputs)
    print(f"Configurations with all workers finished: {completed}/12; pending: {len(state['pending'])}; completion requires audit")
    for gpu, job in state['running'].items():
        log = research / (job['name'] + '_full.log')
        lines = log.read_text().splitlines() if log.exists() else []
        line = lines[-1] if lines else job['name']
        live = verify_process(job['pid'], 'mydata_bench.addbase_eval.run')
        print(f"GPU {gpu} PID {job['pid']} live_verified={live}: {line}")
        match = re.fullmatch(r'(\w+)/(\w+) (\S+) (\d+)/(\d+) ok=(\d+)/(\d+)', line)
        if match is None or match[6] == match[7]:
            continue
        model, protocol, condition = match[1], match[2], match[3]
        path = OUT / f'{model}_{protocol}' / 'predictions' / (condition.replace(':', '_') + '.jsonl')
        try:
            with path.open() as f:
                recent = deque(f, maxlen=int(match[7]))
            rows = [json.loads(row) for row in recent]
            counts = Counter((row['status'], row.get('error') or row.get('parse_error'))
                             for row in rows if row['status'] != 'ok')
            print('  Latest output tail invalid:', dict(counts))
        except (OSError, json.JSONDecodeError) as exc:
            print(f'  Output is being appended; inspect next snapshot: {type(exc).__name__}')
    if (research / 'scheduler_completion.json').exists():
        print('SCHEDULER_FINISHED')


if __name__ == '__main__':
    main()
