"""Release gate for components withheld until the paper is accepted."""
from __future__ import annotations

from typing import NoReturn

NOTICE = (
    "LAMP-Rec reference implementation is not public yet. The full source of "
    "this component will be released in this repository once the paper is "
    "accepted; the public release currently ships the interface and "
    "documentation only."
)


class NotReleasedError(NotImplementedError):
    """Raised when a withheld component is called in the public skeleton."""


def withheld(component: str) -> NoReturn:
    """Print the release notice and raise :class:`NotReleasedError`."""
    print(NOTICE)
    raise NotReleasedError(f"{component}: {NOTICE}")
