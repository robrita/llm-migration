
# Code Quality Standards (Ruff)

## Formatting Rules

- **Line length**: 100 chars (enforced in `pyproject.toml`)
- **Quotes**: Double quotes only
- **Target**: Python 3.11+
- **Pre-commit**: Run `make format` before every commit
- **Cross-platform**: Ensure all code works on both Windows and Linux (use `pathlib.Path`, avoid shell-specific commands)

## Per-File Ignores

- `__init__.py`: Ignores `F401` (unused imports are acceptable for package exports)
- `schemas/gpt_schema.py`: Ignores `N815` (camelCase allowed to match JSON field names)

## Development Commands

```bash
make lint       # Check code style (non-destructive)
make format     # Auto-fix + format (idempotent, safe to run repeatedly)
```

## Best Practices

1. Always run `make format` before committing
2. Fix lint errors before running the app (enforced by `make check-and-run`)
3. Keep code idiomatic to Python 3.11+ (use type hints, modern syntax)
4. Maintain consistency with existing codebase patterns

---

# Documentation notes

- Do not create a new markdown file for summary documentation on new features.
- Write the concise documentation by updating the README.md file instead.

---

# Streamlit Development Guidelines

## Chart Width Configuration

### General Streamlit Charts
The `use_container_width` parameter is deprecated. Use the `width` parameter instead:

| Deprecated | Modern Equivalent |
|------------|-------------------|
| `use_container_width=True` | `width="stretch"` |
| `use_container_width=False` | `width="content"` |

**Correct:**
```python
st.dataframe(df, width="stretch")      # Full container width
st.line_chart(data, width="stretch")
st.bar_chart(data, width="content")    # Fit to content
```

**Incorrect (deprecated):**
```python
st.dataframe(df, use_container_width=True)
st.line_chart(data, use_container_width=True)
st.bar_chart(data, use_container_width=False)
```

### Plotly Charts (st.plotly_chart)
For `st.plotly_chart()` specifically, use the `config` parameter to specify Plotly configuration options instead of width parameters.

**Correct:**
```python
st.plotly_chart(fig, config={"responsive": True})
```

**Incorrect:**
```python
st.plotly_chart(fig, width="stretch")  # Deprecated
st.plotly_chart(fig, use_container_width=True)  # Deprecated
```

### Why This Matters
- `use_container_width` parameter is deprecated across Streamlit components
- `width="stretch"` replaces `use_container_width=True` (full container width)
- `width="content"` replaces `use_container_width=False` (fit to content)
- `st.plotly_chart()` uses Plotly's native configuration system via the `config` parameter
- Following these guidelines avoids deprecation warnings and ensures future compatibility

## Semantic Headers with st.subheader()

Use `st.subheader()` for all section headers instead of `st.markdown("### XXXXXXX")`.

**Correct:**
```python
st.subheader("Section Title")
st.subheader("Field Configuration")
```

**Incorrect (deprecated):**
```python
st.markdown("### Section Title")
st.markdown("### Field Configuration")
```

### Why This Matters
- `st.subheader()` is the semantic Streamlit component designed for headers
- `st.markdown("### ...")` is a workaround that doesn't leverage Streamlit's styling system
- Using semantic components ensures consistency, better accessibility, and proper theme support
- Streamlit components adapt to your app's configured theme automatically

---

# VERY IMPORTANT: Do not duplicate instructions

- Do not reuse, copy or duplicate the instructions above here in AGENTS.md for copilot-instructions.md

---

# Project Best Practices (Learned)

## Navigation & Imports
- Use `helpers/utils.py` for shared helpers (e.g., `render_sidebar`). Import directly: `from helpers.utils import render_sidebar`.
- Use `helpers/llm_eval.py` for LLM evaluation utilities. Import directly: `from helpers.llm_eval import run_parallel_evaluations`.
- Keep sidebar links accurate and minimal: `app.py`, `pages/pg1_Pricing.py`.
- Ensure `pages/` files are named and placed correctly for Streamlit multi-page discovery.
- Call `st.set_page_config(...)` once and before any other Streamlit calls (centralize in `render_sidebar` or the main app).

## Data & Config
- Externalize static data into `inputs/*.json` and load via a cached loader (`@lru_cache`) with validation and logging.
- Keep LLM model configurations in `.conf/llm_eval_config.json` (gitignored), never hardcode API keys.

## Observability
- Use module-level loggers (`logger = logging.getLogger(__name__)`).
- Log key actions and outcomes (data loads, filter counts, preview success/failure).

## Typing & Lint
- Prefer built-in generics (`list`, `dict`) over `typing.List`/`typing.Dict`.
- Run `make lint` (Ruff) and `make format` before commits.

---

# Git Operations Requiring Explicit Approval

**CRITICAL**: The following Git operations must NEVER be performed without explicit human approval:

- `git branch -d` / `git branch -D` (delete local branch)
- `git push origin --delete` (delete remote branch)
- `git reset --hard` (discard commits)
- `git push --force` / `git push -f` (force push)
- `git rebase` on shared branches

Even after a successful merge, always ask before deleting feature branches.

---

# Dependency Management

- Add packages via `uv add <package>` (updates `pyproject.toml` and `uv.lock`)
- Remove packages via `uv remove <package>`
- After adding/removing, run `uv sync` to update environment
- Pin versions for production stability when needed

---

# Environment Variables

- Document all new env vars in `.env.example` with placeholder values
- Never commit actual credentials to `.env` (gitignored)
- Use descriptive names: `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`
- Group related vars with comments in `.env.example`

---

# Breaking Changes

When making breaking changes to APIs or data schemas:

- Bump version in `pyproject.toml` if applicable
- Document migration steps in commit message or PR description
- Update affected tests before merging
- Notify team members if shared interfaces change

---

# Quality Gate Checklist (Before Merge)

Complete these steps after any feature change or update, and before merging:

1. **Code Quality**
   - [ ] `make format` passes (no changes)
   - [ ] `make lint` passes (no errors)

2. **Tests**
   - [ ] `make test-unit` passes
   - [ ] New code has corresponding tests

3. **Documentation**
   - [ ] README.md updated if user-facing changes
   - [ ] Docstrings added for new public functions

4. **Git Hygiene**
   - [ ] Commit messages follow conventional format (`feat:`, `fix:`, `docs:`)
   - [ ] No secrets/credentials committed

5. **Review**
   - [ ] Self-review of diff before commit
   - [ ] Run app locally to verify functionality
