import logging
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
import sys


def setup_logging(outdir: str, prefix: str, silent: bool):
    """Configures multi-destination logging for Panthera."""

    # Create the output directory if it doesn't exist
    log_dir = Path(outdir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{prefix}_run.log"

    # 1. Define Formatters
    # Console is brief; File is detailed with timestamps and line numbers
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )

    # 2. Setup File Handler (Always logs everything at DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # 3. Setup Console Handler
    console_handler = RichHandler(console=Console(file=sys.stdout))
    console_handler.setFormatter(console_formatter)

    # If silent is True, we only show CRITICAL errors to the console
    if silent:
        console_handler.setLevel(logging.CRITICAL)
    else:
        console_handler.setLevel(logging.INFO)

    # 4. Configure Root Logger
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.DEBUG)  # Capture everything internally
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized. Full traces saved to: {log_file}")
