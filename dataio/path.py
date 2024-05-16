"""Path module."""


def rmtree(path):
    """Remove directory."""
    if path.exists():
        for p in path.iterdir():
            if p.is_file():
                p.unlink()
            else:
                rmtree(p)
        path.rmdir()
