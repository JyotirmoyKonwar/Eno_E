from darkguard_trainer.parser_utils import parse_action_text


def test_parse_json_action() -> None:
    parsed = parse_action_text('{"action_type":"inspect","target_id":"x"}')
    assert parsed["action_type"] == "inspect"
    assert parsed["target_id"] == "x"


def test_parse_fallback_action() -> None:
    parsed = parse_action_text("ACTION: click | TARGET: accept_all")
    assert parsed["action_type"] == "click"
    assert parsed["target_id"] == "accept_all"
