
# janus/__main__.py
"""
Unified entry point for the Janus framework.
Supports both physics discovery and AI interpretability research.
"""

import sys
import logging
from pathlib import Path
from janus.cli.main import cli


# Add project root to path for development
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from janus.cli.main import cli

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run CLI
    cli()
