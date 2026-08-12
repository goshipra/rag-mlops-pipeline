# Virtual Environments

A virtual environment is an isolated Python installation that keeps a
project's dependencies separate from your system Python and from other
projects. Without one, installing package A for project 1 can silently
break project 2 if they need incompatible versions of the same library.

The standard-library way to create one is the `venv` module, which ships
with every Python 3 install:

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

Once activated, `pip install <package>` installs only into `.venv`, and
`python`/`pip` inside that shell resolve to the environment's own
interpreter and site-packages, not the system ones. Deactivate with the
`deactivate` command.

Alternatives you will see in the wild:
- **conda / mamba** — manages non-Python dependencies (C libraries, CUDA
  toolkits) as well as Python packages, popular in data science and ML
  because many ML wheels need matching system libraries.
- **virtualenv** — a faster, more feature-rich third-party predecessor to
  `venv` that still sees use in older codebases and CI images.
- **uv** — a newer, Rust-based tool that creates virtual environments and
  resolves/installs dependencies dramatically faster than plain `pip`.

A good rule of thumb: never `pip install` directly into your system
Python. Always work inside a virtual environment (or a container, which is
effectively an isolated environment at the OS level) so a broken
dependency in one project can never take down another.
