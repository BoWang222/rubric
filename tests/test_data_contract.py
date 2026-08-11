from rubric_dpo.data.tokenization import _truncate
from rubric_dpo.data.ultrafeedback import _parse_ratings


def test_rating_contract_rejects_na_and_out_of_range():
    assert _parse_ratings(("1", "2", "3", "5")) == (1.0, 2.0, 3.0, 5.0)
    assert _parse_ratings(("1", "N/A", "3", "5")) is None
    assert _parse_ratings(("1", "2", "6", "5")) is None


def test_token_truncation_keeps_prompt_end_and_single_eos():
    prompt = list(range(1100))
    completion = list(range(2000, 3100)) + [151645]
    p, c, pt, ct = _truncate(prompt, completion)
    assert pt and ct
    assert p == prompt[-1024:]
    assert len(c) == 1024
    assert c[-1] == 151645
    assert c.count(151645) == 1
