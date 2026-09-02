"""Shared contracts and clients.

Every service depends on this package and on nothing else in the system.
Cross-service communication goes through the models in `contracts`; a change
that breaks one fails CI rather than production.
"""

__version__ = "0.1.0"
