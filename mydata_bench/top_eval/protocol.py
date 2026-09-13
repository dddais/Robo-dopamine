"""Load only the official pure prompt/composite definitions (no vLLM imports)."""
import ast
from decimal import Decimal
import math
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


def format_percentage(value):
    """Preserve parsed answer text; never reconstruct feedback from p*100."""
    number = Decimal(str(value))
    if not number.is_finite():
        raise ValueError('Previous SOLE percentage must be finite')
    if isinstance(value, str):
        return value.strip()
    if number == 0:
        return '0'
    text = format(number, 'f')
    return text.rstrip('0').rstrip('.') if '.' in text else text


def parse_progress(raw):
    import re
    tags = re.findall(r'<answer>(.*?)</answer>', raw, flags=re.DOTALL)
    match = re.fullmatch(r'\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*', tags[0]) if len(tags) == 1 else None
    if raw.count('<answer>') != 1 or raw.count('</answer>') != 1 or match is None:
        return {'status': 'parse_error', 'progress': None, 'parse_error': 'Expected exactly one complete numeric answer tag'}
    percentage_text = match.group(1)
    percentage = Decimal(percentage_text)
    value = float(percentage)
    if not math.isfinite(value):
        return {'status': 'parse_error', 'progress': None, 'parse_error': 'Non-finite numeric answer'}
    clipped = max(Decimal('-100'), min(Decimal('100'), percentage))
    return {'status': 'ok', 'raw_percentage': value,
            'raw_percentage_text': percentage_text,
            'percentage_text': percentage_text if clipped == percentage else format_percentage(clipped),
            'progress': float(clipped / 100), 'percentage_clipped': clipped != percentage}
