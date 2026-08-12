# Dependency Pinning and Lock Files

"Pinning" means fixing a dependency to an exact version (`requests==2.31.0`)
instead of an open range (`requests>=2.0`). Pinning matters because an open
range can resolve to a different, newer version every time you build —
which means a build that worked yesterday can fail today even though you
changed nothing, simply because a transitive dependency shipped a breaking
release. This is often called "dependency drift."

There are two related but distinct files worth knowing:

- **`requirements.txt`** (unpinned or loosely pinned) usually lists the
  *direct* dependencies a project needs, often with minimum versions, e.g.
  `flask>=3.0`. It expresses intent, not an exact reproducible build.
- **A lock file** (`requirements.lock`, `poetry.lock`, `uv.lock`,
  `Pipfile.lock`) records the *exact* resolved version of every direct and
  transitive dependency, typically with content hashes. Installing from a
  lock file gives you a byte-for-byte reproducible environment on any
  machine.

The standard workflow is: declare loose ranges in `pyproject.toml` (what
your code *needs*), then generate a lock file (what your code *actually
gets*) and commit the lock file to version control. CI and production
deployments install from the lock file so every environment — a
developer's laptop, the CI runner, and production — resolves to the exact
same dependency graph.

For services that need strict reproducibility, such as an ML inference
service where a subtly different `numpy` or `torch` version can change
numerical output, pinning via a lock file is not optional — it is a
correctness requirement, not just a convenience.
