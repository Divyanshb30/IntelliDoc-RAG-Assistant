"""Regex-based security vulnerability scanner for Python source.

Detects common security anti-patterns: hardcoded secrets, injection sinks,
weak cryptography, dangerous builtins, and insecure deserialization.  This is
a lightweight linter — not a substitute for a full SAST tool — but catches the
issues that most often appear in application code.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Vulnerability:
    """A single detected vulnerability."""

    type: str
    severity: str  # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    line: int
    description: str
    fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "severity": self.severity,
            "line": self.line,
            "description": self.description,
            "fix": self.fix,
        }


@dataclass
class ScanResult:
    """Result of a security scan."""

    success: bool
    vulnerabilities: list[Vulnerability] = field(default_factory=list)
    risk_level: str = "LOW"
    error: str = ""

    @property
    def total_issues(self) -> int:
        return len(self.vulnerabilities)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "risk_level": self.risk_level,
            "total_issues": self.total_issues,
            "error": self.error,
        }


class SecurityScanner:
    """Scan Python source for security vulnerabilities using regex patterns."""

    # Hardcoded-secret patterns → label
    _SECRET_PATTERNS: dict[str, str] = {
        r'password\s*=\s*["\'][^"\']+["\']': "Hardcoded password",
        r'api[_-]?key\s*=\s*["\'][^"\']+["\']': "Hardcoded API key",
        r'secret[_-]?key\s*=\s*["\'][^"\']+["\']': "Hardcoded secret key",
        r'token\s*=\s*["\'][^"\']+["\']': "Hardcoded token",
        r'aws_access_key_id\s*=\s*["\'][^"\']+["\']': "Hardcoded AWS key",
    }

    _SQL_INJECTION_PATTERNS: list[str] = [
        r'execute\s*\(\s*["\'].*%s.*["\']\s*%',
        r'execute\s*\(\s*f["\'].*\{.*\}.*["\']',
        r'execute\s*\(\s*["\'].*["\']\s*\+',
    ]

    _WEAK_HASHES: list[str] = ["md5", "sha1"]

    _DANGEROUS_FUNCS: dict[str, tuple[str, str]] = {
        "eval(": ("Code Injection", "Arbitrary code execution via eval()"),
        "exec(": ("Code Injection", "Arbitrary code execution via exec()"),
        "os.system(": ("Command Injection", "Shell command injection via os.system()"),
    }

    _IP_PATTERN = re.compile(r'["\'](?:\d{1,3}\.){3}\d{1,3}["\']')

    # ── Public API ───────────────────────────────────────────────────────

    def scan(self, code: str) -> ScanResult:
        """Scan *code* and return a :class:`ScanResult`.

        Args:
            code: Python source to scan.

        Returns:
            The scan result with any vulnerabilities and an overall risk level.
        """
        vulns: list[Vulnerability] = []
        lines = code.splitlines()

        vulns.extend(self._scan_secrets(lines))
        vulns.extend(self._scan_sql_injection(lines))
        vulns.extend(self._scan_weak_crypto(lines))
        vulns.extend(self._scan_dangerous_functions(lines))
        vulns.extend(self._scan_deserialization(lines))
        vulns.extend(self._scan_shell_true(lines))
        vulns.extend(self._scan_yaml_load(lines))
        vulns.extend(self._scan_hardcoded_ip(lines))

        return ScanResult(
            success=True,
            vulnerabilities=vulns,
            risk_level=self._risk_level(vulns),
        )

    # ── Individual scans ─────────────────────────────────────────────────

    def _scan_secrets(self, lines: list[str]) -> list[Vulnerability]:
        out: list[Vulnerability] = []
        for n, line in enumerate(lines, 1):
            for pattern, label in self._SECRET_PATTERNS.items():
                if re.search(pattern, line, re.IGNORECASE):
                    out.append(
                        Vulnerability(
                            type=label,
                            severity="CRITICAL",
                            line=n,
                            description=f"Hardcoded credential: {line.strip()[:50]}…",
                            fix="Load secrets from environment variables or a secrets manager",
                        )
                    )
        return out

    def _scan_sql_injection(self, lines: list[str]) -> list[Vulnerability]:
        out: list[Vulnerability] = []
        for n, line in enumerate(lines, 1):
            for pattern in self._SQL_INJECTION_PATTERNS:
                if re.search(pattern, line):
                    out.append(
                        Vulnerability(
                            type="SQL Injection",
                            severity="CRITICAL",
                            line=n,
                            description="String-formatted SQL is vulnerable to injection",
                            fix="Use parameterised queries or an ORM",
                        )
                    )
                    break
        return out

    def _scan_weak_crypto(self, lines: list[str]) -> list[Vulnerability]:
        out: list[Vulnerability] = []
        for n, line in enumerate(lines, 1):
            for algo in self._WEAK_HASHES:
                if f"hashlib.{algo}" in line:
                    out.append(
                        Vulnerability(
                            type="Weak Cryptography",
                            severity="HIGH",
                            line=n,
                            description=f"Weak hash algorithm: {algo.upper()}",
                            fix="Use SHA-256+ for integrity or bcrypt/argon2 for passwords",
                        )
                    )
        return out

    def _scan_dangerous_functions(self, lines: list[str]) -> list[Vulnerability]:
        out: list[Vulnerability] = []
        for n, line in enumerate(lines, 1):
            for func, (vuln_type, desc) in self._DANGEROUS_FUNCS.items():
                if func in line:
                    out.append(
                        Vulnerability(
                            type=vuln_type,
                            severity="CRITICAL",
                            line=n,
                            description=desc,
                            fix=f"Avoid {func.rstrip('(')} with untrusted input",
                        )
                    )
        return out

    def _scan_deserialization(self, lines: list[str]) -> list[Vulnerability]:
        out: list[Vulnerability] = []
        for n, line in enumerate(lines, 1):
            if "pickle.load" in line:
                out.append(
                    Vulnerability(
                        type="Insecure Deserialization",
                        severity="HIGH",
                        line=n,
                        description="pickle can execute arbitrary code during load",
                        fix="Use JSON or another safe serialisation format",
                    )
                )
        return out

    def _scan_shell_true(self, lines: list[str]) -> list[Vulnerability]:
        out: list[Vulnerability] = []
        for n, line in enumerate(lines, 1):
            if re.search(r"subprocess\.(run|call|Popen|check_output)", line) and "shell=True" in line:
                out.append(
                    Vulnerability(
                        type="Command Injection",
                        severity="HIGH",
                        line=n,
                        description="subprocess with shell=True enables shell injection",
                        fix="Pass args as a list and use shell=False (the default)",
                    )
                )
        return out

    def _scan_yaml_load(self, lines: list[str]) -> list[Vulnerability]:
        out: list[Vulnerability] = []
        for n, line in enumerate(lines, 1):
            if "yaml.load(" in line and "Loader=" not in line and "SafeLoader" not in line:
                out.append(
                    Vulnerability(
                        type="Insecure Deserialization",
                        severity="HIGH",
                        line=n,
                        description="yaml.load without SafeLoader can execute arbitrary code",
                        fix="Use yaml.safe_load() or Loader=yaml.SafeLoader",
                    )
                )
        return out

    def _scan_hardcoded_ip(self, lines: list[str]) -> list[Vulnerability]:
        out: list[Vulnerability] = []
        for n, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            match = self._IP_PATTERN.search(line)
            if match and "127.0.0.1" not in match.group() and "0.0.0.0" not in match.group():
                out.append(
                    Vulnerability(
                        type="Hardcoded IP address",
                        severity="LOW",
                        line=n,
                        description=f"Hardcoded IP {match.group()} reduces portability and can leak infra details",
                        fix="Move host addresses to configuration",
                    )
                )
        return out

    # ── Risk aggregation ─────────────────────────────────────────────────

    @staticmethod
    def _risk_level(vulns: list[Vulnerability]) -> str:
        """Roll individual severities up into an overall risk level."""
        if any(v.severity == "CRITICAL" for v in vulns):
            return "CRITICAL"
        if any(v.severity == "HIGH" for v in vulns):
            return "HIGH"
        if vulns:
            return "MEDIUM"
        return "LOW"
