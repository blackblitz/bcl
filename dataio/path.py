"""Path module."""

from pathlib import Path


def clear(path):
    """Clear files and directories under a directory."""
    path = Path(path)
    if path.exists():
        if not path.is_dir():
            raise ValueError('path is not of a directory')
        for p in path.iterdir():
            if p.is_file():
                p.unlink()
            else:
                clear(p)
                p.rmdir()
    else:
        path.mkdir(parents=True)
