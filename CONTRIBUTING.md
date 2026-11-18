# Contributing to Helios

## Development flow

1. Create a feature branch from `main`
2. Make focused changes (small PRs are preferred)
3. Run local checks before opening a PR
4. Open PR with clear description and testing notes

## Local checks

Backend:

```bash
python -m pip install -e .
python -m pytest
```

Frontend:

```bash
cd frontend
npm run build
```

## Code style

- Keep functions and modules focused
- Avoid unrelated refactors in the same PR
- Update docs when behavior changes
- Keep configuration changes explicit and documented

## Commit message guideline

Use concise, scoped messages, for example:

- `feat(api): add molecule visualization endpoint`
- `fix(startup): handle missing node_modules`
- `docs(setup): add windows startup steps`

## Reporting issues

Include:

- OS and Python/Node versions
- Exact command run
- Full error output
- Steps to reproduce
