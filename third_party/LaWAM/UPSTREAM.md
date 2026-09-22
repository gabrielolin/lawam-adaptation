# LaWAM upstream provenance

- Upstream repository: https://github.com/RLinf/LaWAM.git
- Local Git remote: `lawam-upstream`
- Imported ref: `main`
- Imported upstream commit: `c12168d9077edea67fe7c1b70d08b80b8e65a9c8`
- Import method: `git subtree add --prefix=third_party/LaWAM lawam-upstream main --squash`
- Import date: 2026-09-21

`third_party/LaWAM` is vendored third-party research code. Do not modify it for project-specific research. Put integrations in `src/lawam_adaptation`; update this file if a future upstream subtree import changes the revision.

To update the subtree intentionally:

```bash
git fetch lawam-upstream main
git subtree pull --prefix=third_party/LaWAM lawam-upstream main --squash
```
