# Wheels vs. Source Distributions (sdists)

When you run `python -m build`, a Python project produces two kinds of
distribution artifacts, and understanding the difference matters for both
publishing and debugging slow installs.

**Source distribution (sdist)**, a `.tar.gz` file, contains your project's
source code plus the metadata needed to build it (`pyproject.toml`, etc.).
Installing from an sdist means pip runs your build backend locally to
produce an installable package — this always works, on any platform, but
can be slow, and it fails if the build requires a compiler or system
libraries the target machine does not have.

**Wheel (`.whl`)**, defined by PEP 427, is a pre-built, ready-to-install
package: a zip archive laid out exactly as it needs to land in
`site-packages`, with no build step required at install time. Installing
from a wheel is just "unzip into place," which is why `pip install` from
a wheel is dramatically faster than from an sdist — especially for
packages with compiled C/C++/Rust extensions like `numpy`, `pydantic`, or
`cryptography`.

Because compiled extensions are platform- and Python-version-specific,
projects with native code publish multiple wheels per release — one per
combination of OS, CPU architecture, and Python ABI (you'll see filenames
like `numpy-1.26.4-cp311-cp311-manylinux_2_17_x86_64.whl`). Pure-Python
projects can ship a single "universal" wheel that works everywhere.

Best practice when publishing a package to PyPI: always upload both a
wheel and an sdist. The wheel gives most users a fast, pre-built install;
the sdist is the fallback for platforms you didn't build a wheel for, and
it is also required by some downstream packagers (e.g., Linux
distributions) that insist on building from source.
