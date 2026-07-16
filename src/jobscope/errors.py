"""Exceptions shared across jobscope."""


class JobscopeError(Exception):
    """A user-facing error; the CLI prints its message and exits non-zero."""
