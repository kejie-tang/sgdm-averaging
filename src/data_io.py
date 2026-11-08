"""Download dataset files without retaining incomplete transfers."""

from __future__ import annotations

import os
import tempfile
import urllib.request


def download_file(url, destination):
    """Replace destination only after a complete download succeeds."""
    parent = os.path.dirname(os.path.abspath(destination))
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=parent)
    os.close(descriptor)
    try:
        urllib.request.urlretrieve(url, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
