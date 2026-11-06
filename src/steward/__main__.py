"""Allow ``python -m steward``."""

import sys

from steward.cli import main

sys.exit(main())
