from dga_fraudulents_dataset.validation import normalize_domain, validate_domain


def test_normalize_and_validate_domain():
    assert normalize_domain(" Example.COM. ") == "example.com"
    ok = validate_domain("Example.COM.")
    assert ok.is_valid is True
    assert ok.normalized == "example.com"

    bad = validate_domain("not_a_domain")
    assert bad.is_valid is False


def test_validation_converts_idn_and_enforces_label_limits():
    idn = validate_domain(" B\u00dcCHER.Example. ")
    assert idn.is_valid is True
    assert idn.normalized == "xn--bcher-kva.example"

    assert validate_domain(f"{'a' * 64}.com").reason == "label_too_long"
    assert validate_domain("-invalid.com").reason == "invalid_character"
    assert validate_domain("example.c1").reason == "invalid_suffix"
