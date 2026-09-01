"""Ground-truth fixture for the AST analyzer accuracy benchmark.

Each anti-pattern is annotated with an ``# EXPECT: <Issue.type>`` marker on the
exact line the analyzer reports it (the ``def``/statement line).  The analyzer
accuracy eval parses these markers to build ground truth, so this file's
markers and the analyzer's output must stay in agreement.

This module is intentionally NOT imported or executed — it only exists to be
parsed.  Do not "fix" the anti-patterns below.
"""

import hashlib  # noqa: F401 — used by an intentional weak-crypto example elsewhere


def append_item(item, bucket=[]):  # EXPECT: Mutable default argument
    """Mutable default argument shared across calls."""
    bucket.append(item)
    return bucket


def swallow_everything():
    """Bare except that hides all errors."""
    try:
        risky = 1 / 0
        return risky
    except:  # EXPECT: Bare except clause
        pass


def over_broad_handler():
    """Catches Exception, which is too broad."""
    try:
        return compute()
    except Exception:  # EXPECT: Broad exception handler
        return None


def has_unused_variable():
    """Assigns a variable that is never read."""
    leftover = compute_total()  # EXPECT: Unused variable
    return 42


def deeply_nested(data):  # EXPECT: Deep nesting
    """Four levels of control-flow nesting."""
    out = []
    for item in data:
        if item > 0:
            if item < 100:
                if item % 2 == 0:
                    out.append(item)
    return out


def tangled_logic(a, b, c, d, e):  # EXPECT: High cyclomatic complexity
    """Many branches → high cyclomatic complexity."""
    total = 0
    if a > 0:
        total += 1
    if b > 0:
        total += 1
    if c > 0 and d > 0:
        total += 1
    if e > 0 or a > b:
        total += 1
    for i in range(a):
        if i % 2 == 0:
            total += i
        elif i % 3 == 0:
            total -= i
    while total > 100:
        total -= 10
    return total


def uses_global():
    """Mutates module-level state via global."""
    global _counter  # EXPECT: Global statement
    _counter += 1


def uses_assert(value):
    """Relies on assert for validation (stripped under -O)."""
    assert value > 0  # EXPECT: Assert in production code
    return value * 2


def long_function(x):  # EXPECT: Long function
    """A function padded past the long-function threshold (>50 lines)."""
    s = x
    s = s + 1
    s = s + 2
    s = s + 3
    s = s + 4
    s = s + 5
    s = s + 6
    s = s + 7
    s = s + 8
    s = s + 9
    s = s + 10
    s = s + 11
    s = s + 12
    s = s + 13
    s = s + 14
    s = s + 15
    s = s + 16
    s = s + 17
    s = s + 18
    s = s + 19
    s = s + 20
    s = s + 21
    s = s + 22
    s = s + 23
    s = s + 24
    s = s + 25
    s = s + 26
    s = s + 27
    s = s + 28
    s = s + 29
    s = s + 30
    s = s + 31
    s = s + 32
    s = s + 33
    s = s + 34
    s = s + 35
    s = s + 36
    s = s + 37
    s = s + 38
    s = s + 39
    s = s + 40
    s = s + 41
    s = s + 42
    s = s + 43
    s = s + 44
    s = s + 45
    s = s + 46
    s = s + 47
    s = s + 48
    s = s + 49
    s = s + 50
    s = s + 51
    s = s + 52
    s = s + 53
    s = s + 54
    s = s + 55
    return s


async def async_mutable(items={}):  # EXPECT: Mutable default argument
    """Async function with a mutable default — exercises AsyncFunctionDef support."""
    items["seen"] = True
    return items


def compute():
    """Helper referenced above; return-hinted to avoid incidental findings."""
    return 1


def compute_total():
    """Helper referenced above."""
    return 0


_counter = 0
