"""AI-isms Scanner: ESLint for prose. Detect AI-isms and low-quality writing patterns."""

__version__ = "0.1.0"

from .engine import analyze  # noqa: E402
from .models import Finding, Report, Span  # noqa: E402

__all__ = ["analyze", "Finding", "Report", "Span", "__version__"]
