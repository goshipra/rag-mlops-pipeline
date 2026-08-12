# Publishing a Package to PyPI

Publishing to the Python Package Index (PyPI) makes a project installable
anywhere with `pip install <your-package>`. The modern, recommended flow
uses two tools: `build` to create the distribution artifacts, and `twine`
to upload them.

```bash
python -m pip install --upgrade build twine
python -m build                 # creates dist/*.whl and dist/*.tar.gz
python -m twine upload dist/*   # uploads both to PyPI
```

Before uploading for real, always test against **TestPyPI**
(`https://test.pypi.org`), a separate instance for practicing the release
process without polluting the real index or permanently claiming a
package name:

```bash
python -m twine upload --repository testpypi dist/*
```

Authentication should use an **API token**, not your PyPI account
password. Generate a scoped token (ideally limited to a single project)
from your PyPI account settings, and pass it via `twine upload -u
__token__ -p <token>` or, better, store it in `~/.pypirc` or a CI secret
so it never appears in shell history.

A critical, easy-to-forget rule: **PyPI version numbers are immutable.**
Once `1.2.3` is uploaded, you cannot re-upload a file under that same
version, even to fix a bug in the release — you must bump the version
(e.g., to `1.2.4`) and upload again. This is why many teams automate
releases entirely from CI, triggered by a Git tag, so version bumps and
uploads always stay in sync and a human never accidentally re-runs a
stale build.
