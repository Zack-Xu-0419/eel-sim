"""Keep readable reports separate from numerical and media outputs."""
from pathlib import Path


from src.paths import ROOT


def report_directory(output_dir):
    """Return Reports/<study> for Outputs/<study>, otherwise output/Reports.

    CSV, JSON, PNG, SVG, and video paths are unaffected. Custom outputs outside
    this project's Outputs tree keep their reports inside that custom folder.
    """
    output = Path(output_dir).resolve()
    try:
        relative = output.relative_to(ROOT / 'Outputs')
    except ValueError:
        destination = output / 'Reports'
    else:
        destination = ROOT / 'Reports' / relative
    destination.mkdir(parents=True, exist_ok=True)
    return destination
