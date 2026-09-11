"""Load only the official pure prompt/composite definitions (no vLLM imports)."""
import ast
from pathlib import Path
import cv2
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / 'addbase_eval/references/rewardgen/rewardgen/sole.py'
NAMES = {'system_prompt_template', 'user_question_template_external_view',
         'resize_with_padding', 'create_composite_frame'}
tree = ast.parse(SOURCE.read_text())
nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in NAMES)
         or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in NAMES for t in n.targets))]
namespace = {'np': np, 'cv2': cv2}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), namespace)
SYSTEM_PROMPT = namespace['system_prompt_template']
OFFICIAL_QUESTION = namespace['user_question_template_external_view']
create_composite_frame = namespace['create_composite_frame']


def parse_progress(raw):
    import re
    matches = re.findall(r'<answer>\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*</answer>', raw)
    if len(matches) != 1:
        return {'status': 'parse_error', 'progress': None, 'parse_error': 'Expected exactly one complete numeric answer tag'}
    value = float(matches[0])
    return {'status': 'ok', 'raw_percentage': value, 'progress': max(-1.0, min(1.0, value/100)),
            'percentage_clipped': not -100 <= value <= 100}
