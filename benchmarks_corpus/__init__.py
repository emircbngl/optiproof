"""Validation corpus — known (slow, fast, trap, no-win) quadruples.

The corpus is how we test the tool against its own philosophy: it must *find +
prove* the real speedups, *reject* the behaviour-changing traps (the most
important guard — they prove the harness can't be fooled), and *not fabricate*
wins where none exist.
"""
