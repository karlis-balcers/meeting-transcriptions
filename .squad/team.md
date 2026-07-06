# Squad Team

> meeting-transcriptions

## Coordinator

| Name | Role | Notes |
|------|------|-------|
| Squad | Coordinator | Routes work, enforces handoffs and reviewer gates. |

## Members

| Name | Role | Charter | Status |
|------|------|---------|--------|
| Danny | Lead Engineer | `.squad/agents/danny/charter.md` | ✅ Active |
| Rusty | Go/Core Developer | `.squad/agents/rusty/charter.md` | ✅ Active |
| Livingston | Platform Developer | `.squad/agents/livingston/charter.md` | ✅ Active |
| Basher | QA Engineer | `.squad/agents/basher/charter.md` | ✅ Active |
| Scribe | Session Logger | `.squad/agents/scribe/charter.md` | 📋 Silent |
| Ralph | Work Monitor | `.squad/agents/ralph/charter.md` | 🔄 Monitor |

## Project Context

- **Project:** meeting-transcriptions
- **Created:** 2026-05-27
- **Current architecture:** Root-level Go module with entrypoints under `cmd/`, packages under `internal/`, documentation under `docs/`, and Windows builds produced by `build_transcribe_win64.bat` into `build/windows-amd64/`.
