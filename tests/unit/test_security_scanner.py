"""Unit tests for the security scanner."""

from __future__ import annotations

import pytest

from intellicode.analysis import SecurityScanner


@pytest.fixture
def scanner() -> SecurityScanner:
    return SecurityScanner()


@pytest.mark.parametrize(
    ("code", "expected_type"),
    [
        ('password = "hunter2"', "Hardcoded password"),
        ('api_key = "sk-abc123"', "Hardcoded API key"),
        ("eval(user_input)", "Code Injection"),
        ("exec(payload)", "Code Injection"),
        ("os.system(cmd)", "Command Injection"),
        ("hashlib.md5(data)", "Weak Cryptography"),
        ("hashlib.sha1(data)", "Weak Cryptography"),
        ("pickle.load(f)", "Insecure Deserialization"),
        ("subprocess.run(cmd, shell=True)", "Command Injection"),
        ("yaml.load(stream)", "Insecure Deserialization"),
    ],
)
def test_detects_vulnerability(scanner, code, expected_type):
    result = scanner.scan(code)
    assert result.success
    assert expected_type in {v.type for v in result.vulnerabilities}


def test_sql_injection_fstring(scanner):
    code = 'cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")'
    result = scanner.scan(code)
    assert "SQL Injection" in {v.type for v in result.vulnerabilities}


def test_risk_level_critical(scanner):
    result = scanner.scan("eval(x)")
    assert result.risk_level == "CRITICAL"


def test_risk_level_high_without_critical(scanner):
    result = scanner.scan("pickle.load(f)")
    assert result.risk_level == "HIGH"


def test_clean_code_no_vulnerabilities(scanner, clean_code_path):
    code = clean_code_path.read_text(encoding="utf-8")
    result = scanner.scan(code)
    assert result.risk_level == "LOW"
    assert result.total_issues == 0


def test_safe_yaml_load_not_flagged(scanner):
    """yaml.safe_load and explicit SafeLoader should not trip the scanner."""
    assert scanner.scan("yaml.safe_load(stream)").total_issues == 0
    assert scanner.scan("yaml.load(stream, Loader=yaml.SafeLoader)").total_issues == 0


def test_localhost_ip_not_flagged(scanner):
    """Loopback / bind-all addresses are not treated as leaked infra."""
    assert scanner.scan('host = "127.0.0.1"').total_issues == 0


def test_line_numbers_are_reported(scanner):
    result = scanner.scan("x = 1\ny = 2\neval(z)")
    vuln = next(v for v in result.vulnerabilities if v.type == "Code Injection")
    assert vuln.line == 3
