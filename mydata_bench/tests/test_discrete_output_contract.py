import pytest

from mydata_bench.roboreward_eval.runner import parse_native_score
from mydata_bench.qwen_eval.protocols import parse_protocol_output, DISCRETE_PROTOCOLS


@pytest.mark.parametrize('text',[
    'ANSWER: 1.5','ANSWER: 5.0','ANSWER: 5/5','ANSWER: 5e0','ANSWER: 5extra',
    'ANSWER: 5\nANSWER: 1','ANSWER: bad\nANSWER: 5','ANSWER: 5\nANSWER:',
    'ANSWER: 0','ANSWER: 6','ANSWER: nan',
])
def test_malformed_or_ambiguous_discrete_answers_do_not_become_valid_rewards(text):
    with pytest.raises(ValueError):parse_native_score(text)
    for protocol in DISCRETE_PROTOCOLS:
        with pytest.raises(ValueError):parse_protocol_output(protocol,text)


@pytest.mark.parametrize('text,expected',[
    ('ANSWER: 1',1),('ANSWER: 5.',5),('**ANSWER: 3**',3),
    ('Final state is correct.\nANSWER: 5',5),('ANSWER: 4\nOne minor requirement is missing.',4),
])
def test_single_documented_discrete_answer_allows_explanatory_text(text,expected):
    assert parse_native_score(text)==expected
