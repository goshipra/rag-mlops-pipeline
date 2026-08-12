# Entry Points and Console Scripts

Entry points are how a Python package tells the packaging system "when
this is installed, expose a command-line tool" (or, more generally, "make
this importable object discoverable by name to other packages"). The most
common kind is a **console script**: a small, auto-generated executable
that, when run, imports a function from your package and calls it.

Declared in `pyproject.toml` under `[project.scripts]`:

```toml
[project.scripts]
rag-ingest = "rag_pipeline.ingest:main"
```

After `pip install .`, this generates an executable named `rag-ingest` on
the user's `PATH` that is equivalent to running
`python -c "from rag_pipeline.ingest import main; main()"`. That's how
tools like `black`, `pytest`, and `pip` itself give you a plain command
instead of requiring `python -m <package>`.

Beyond console scripts, the broader **entry points** mechanism
(`[project.entry-points."some.group"]`) is a plugin system: a package can
declare that it provides an implementation for a named "group," and any
other package can discover all installed implementations at runtime via
`importlib.metadata.entry_points(group="some.group")`. This is how tools
like `pytest` and `mkdocs` support third-party plugins without needing to
import them by name in advance — the plugin registers itself at install
time, and the host application discovers it dynamically.

Console scripts are the simplest and most common entry point group in
practice: any CLI-driven Python package — including data or ML pipeline
tools — typically exposes its `ingest`, `train`, or `serve` commands this
way instead of asking users to remember module paths.
