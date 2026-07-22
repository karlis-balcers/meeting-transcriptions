# Cross-Platform Test Plan — Stand-Alone Terminal App

**Status:** Draft (pre-implementation) — *Authored by Basher (QA), 2026-07-03*
**Goal under test:** *"A stand-alone app (no UI, terminal-only) that works on Mac, Windows, and Linux, with no Windows Python helper dependency."*
**Architecture change tracked:** Removal of the Windows `ExternalRecorder`/Python WASAPI path (`windows_helper.go`, `TRANSCRIBE_WINDOWS_PYTHON`, `TRANSCRIBE_WINDOWS_AUDIO_HELPER`) in favor of a unified **ffmpeg-backed recorder on all three OSes**.

> **2026-07-06 note:** This document is historical for the Path-C/ffmpeg-unified gate. New Windows recorder-sidecar / WASAPI-loopback QA coverage is tracked in `windows-wasapi-sidecar-test-plan.md` and supersedes the Windows-output assumptions below.

> ## Conventions this plan follows
> The existing `transcribe/internal/` tests use the **Go standard `testing` package**, are **table-driven where applicable**, and rely on **injectable seams** (`audio.Discoverer`, `audio.Recorder`, `app.Dependencies`, `tui.Controller`) rather than real hardware. Following the squad `test-discipline` skill, every API change must land with updated tests in the same commit, and any assertion that references the removed Python helper must be deleted/rewritten in lockstep. The `squad-conventions` skill backs the principle: **cross-platform code must pass on all platforms** (it documents a Node runner, but the invariant is identical here).

---

## Legend

| Mark | Meaning |
| --- | --- |
| ✅ **CAN-WRITE-NOW** | Test can be drafted today against the *current* code; tests pure, side-effect-free logic that survives the refactor. |
| 📋 **DOC-ONLY** | Documentation/contract note, not an automated test (CI build matrix steps, runbook, manual TTY checks). |
| ❓ **DESIGN-DEPENDENT** | Blocked on Danny's architecture decision (e.g. what *exactly* replaces the helper; how no-TTY is reported). Plan the scenario now, write the assertion after the decision lands. |

---

## A. Build Matrix Verification — 📋 DOC-ONLY (CI gate)

The `transcribe/README.md` already documents the six cross-compile smoke checks. The plan: make that matrix a **gated CI step** that must run with `CGO_ENABLED=0`, mirroring what `build_transcribe_win64.bat` already does for the Windows amd64 target.

**Scenarios (CI build script — `go build` for each target):**

```
CGO_ENABLED=0 GOOS=linux   GOARCH=amd64 go build ./cmd/transcribe
CGO_ENABLED=0 GOOS=linux   GOARCH=arm64 go build ./cmd/transcribe
CGO_ENABLED=0 GOOS=darwin  GOARCH=amd64 go build ./cmd/transcribe
CGO_ENABLED=0 GOOS=darwin  GOARCH=arm64 go build ./cmd/transcribe
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build ./cmd/transcribe
CGO_ENABLED=0 GOOS=windows GOARCH=arm64 go build ./cmd/transcribe
```

**Acceptance criteria (document these in the runbook):**
- All six targets compile with zero CGO — the binary has no C/cgo dependency on any OS, so it is genuinely self-contained.
- Windows arm64 is no longer special-cased through WSL just to stage a Python `.venv`; once the helper is removed, `build_transcribe_win64.bat` must drop its `.venv`/`audio_capture.py` staging steps (regression watch — see §H).

**Why DOC-ONLY:** this is a CI/build-level concern, not a `go test` unit. It belongs as a Make target or GitHub Action, with the recipe documented here for Livingston.

---

## B. Audio Device Discovery per OS — ✅ CAN-WRITE-NOW (graceful failure) + 📋 DOC-ONLY (live-device runbook)

`SystemDiscoverer.ListDevices` already branches on `runtime.GOOS` (`linux`→pactl, `darwin`→AVFoundation defaults, `windows`→ffmpeg DShow + PowerShell endpoint fallback). The cross-platform invariant to lock in: **discovery must never crash when ffmpeg / devices are absent — it returns devices-or-warnings-or-error.**

### B.1 — Discovery degrades gracefully when ffmpeg/the device daemon is missing — ✅ CAN-WRITE-NOW (Linux, table-driven)

The current `listLinux` already demonstrates the contract: when `pactl` is missing it synthesizes a default mic device and emits a warning instead of erroring. Generalize this expectation as a table-driven test in `internal/audio/discover_test.go`.

```
TestLinuxDiscoveryDegradesGracefullyWhenPactlAbsent
  - Given: a fake exec/runner that returns pactl-not-found
  - Expect: ListDevices returns (>=1 mic device, [warning mentioning pactl], nil-error)
  - Hard rule: err MUST be nil here — only empty-list+empty-warning is a hard failure.
```

> **Note:** `SystemDiscoverer` shells out via `exec.CommandContext` directly, so injecting a fake requires either **(a)** a small refactor to accept an `Exec` seam, or **(b)** running the assertion in a `t.Setenv`/PATH-scrubbed environment. Flag (a) to Livingston as the preferred seam during the refactor — it also enables testing the Windows path on Linux/CI runners. **❓ DESIGN-DEPENDENT on whether the refactor introduces an exec seam.** Either way the *contract* (warning, not panic) is testable now on the Linux path as-is via PATH manipulation.

### B.2 — Unknown/unsupported OS returns configured devices + warning, not a crash — ✅ CAN-WRITE-NOW

The `default` branch of `ListDevices` returns `configuredDevices(...)` plus a `"not implemented"` warning. Add a test that pins this (it's the safety net for exotic platforms like FreeBSD). Can be written today; it's pure logic.

### B.3 — Windows discovery without ffmpeg returns a clear error — ✅ CAN-WRITE-NOW (after exec seam) / ❓ DESIGN-DEPENDENT today

Currently `listWindows` → `ffmpegPath()` returns `"ffmpeg was not found in PATH..."`. Pin that the error is **non-nil and mentions ffmpeg** so a missing binary produces a discoverable message rather than an empty device list + silent later failure. Blocked on the exec seam (B.1) for cross-platform CI; on a real Windows host it's runnable now.

### B.4 — Live-device discovery smoke test — 📋 DOC-ONLY runbook

Manual checklist (one row per OS) for the QA engineer to confirm on real hardware: plug a USB mic, run `transcribe --list-devices`, and assert that both a mic and a loopback/monitor output surface. Document expected ffmpeg/`pactl`/AVFoundation invocation per OS.

---

## C. Recorder Selection — ffmpeg Everywhere vs. Removed Python Helper — ✅ CAN-WRITE-NOW (contract) + ❓ DESIGN-DEPENDENT (new selection logic)

This is the core of the refactor. Today `ffmpeg.go` ships a single `ExternalRecorder` whose `commandForRequest` branches: **Windows output → Python helper; everything else → ffmpeg.** After Danny/Livingston/Rusty's work, the Python branch must be gone and all three OSes route through ffmpeg.

### C.1 — ffmpeg input args pick the correct backend per OS — ✅ CAN-WRITE-NOW (table-driven, pure logic)

`inputArgs` is pure and already unit-tested for dshow; extend it to a **table** covering all platforms in `ffmpeg_test.go`:

```
TestInputArgsSelectBackendPerPlatform (table-driven)
  {linux,   "",                   "pulse"}      → ["-f","pulse","-i", id]
  {darwin,  "",                   "avfoundation"} → ["-f","avfoundation","-i", id]
  {windows, "",                   "dshow"}       → ["-f","dshow","-i","audio="+id]
  {any,     "explicit-backend",   "..."}          → backend wins over runtime.GOOS
  {windows, device.Backend="dshow", loopback}     → raw render name, NO "[Loopback]" suffix (existing TestWindowsLoopbackOutputArgsUseRawRenderName)
```

This pins the cross-platform ffmpeg contract and can be committed today — it doesn't reference the helper at all.

### C.2 — Windows output capture must NOT route through a Python helper — ❓ DESIGN-DEPENDENT (write after refactor)

After the removal, `commandForRequest` for `windows + output` must produce a pure-ffmpeg command (the same shape as mic capture, no `python`, no `audio_capture.py`, no `record-output`). Draft the assertion shape now:

```
TestWindowsOutputCaptureUsesFfmpegNotPythonHelper
  - recorder := <New recorder after refactor> // ExternalRecorder or its replacement
  - cmd, args, err := recorder.commandForRequest(ChunkRequest{Source:SourceOutput, Device: <render endpoint>, Platform:"windows", ...}, file)
  - Expect: err == nil
  - Expect: cmd does NOT contain "python" and args joined does NOT contain
            "audio_capture.py", "record-output", "--device-id"
  - Expect: args contain "-f","dshow" (or whatever Rusty's unified backend chooses)
```

Marked ❓ because the exact recorder API (does `ExternalRecorder` lose `PythonPath`/`HelperScriptPath`? does it get renamed `FfmpegRecorder`?) is Danny's call.

### C.3 — Recorder validation does not probe for a Python interpreter — ❓ DESIGN-DEPENDENT

The current `Validate` (ffmpeg.go ~line 45) for `windows + output` calls `commandForRequest`, which under the hood needs `windowsPythonPath`. After removal, `Validate` must only check ffmpeg presence. The test asserts `Validate` succeeds with ffmpeg available and **fails only with an ffmpeg-centric message** — never one mentioning Python.

---

## D. Config Loading — ✅ CAN-WRITE-NOW (identical cross-platform)

`config.Load(path, explicit, env)` and `config.RuntimeOverrides` are **already OS-agnostic** (proven by `config_test.go`: precedence config > env > overrides, defaults, invalid-env fallback). The cross-platform invariant to lock: **`config.Load` produces byte-identical results for the same inputs across all OSes.**

### D.1 — Config precedence/overrides are OS-independent — ✅ CAN-WRITE-NOW

Extend `config_test.go` (or add a `config_cross_platform_test.go`) that runs the existing precedence scenarios but parametrizes over a representative path separator set. The logic already uses `filepath`; assert that it does **not** hardcode `/` or `\`.

### D.2 — Default config path resolves per-OS via `~/.transcribe/config.yaml` — ✅ CAN-WRITE-NOW

`config.DefaultPath()` uses `os.UserHomeDir()`. Pin a test that for a scrubbed `HOME`/`USERPROFILE`, it returns a path ending in `.transcribe/config.yaml` (using `filepath.Join`, no separator literals). This is the same Windows-safety rule the `squad-conventions` skill enforces.

### D.3 — `ApplyRuntimeOverrides` does not carry Python-helper keys — ❓ DESIGN-DEPENDENT

If the refactor introduces new CLI flags (e.g. `--ffmpeg-path`), or removes any, update `RuntimeOverrides` and its test in the same commit per the **test-discipline** rule. Plan the scenario; the field list is design-dependent.

---

## E. `--silent` Mode and TUI Mode — ✅ CAN-WRITE-NOW (silent) + ❓ DESIGN-DEPENDENT (TUI no-TTY)

### E.1 — `--silent` writes the final transcript to stdout after interrupt — ✅ CAN-WRITE-NOW

`app.RunSilent` already exists and is fully injectable. It appends device/transcript info to **stderr** and the rendered transcript to **stdout**, blocking on `<-ctx.Done()`. Draft the test in `internal/app/app_test.go` (matches existing style with a `blockingRecorder` / fake transcriber):

```
TestRunSilentWritesTranscriptToStdoutAndMetricsToStderr
  - Build a Prepared with a fake recorder + cancelAfterTranscripts
  - Capture stdout/stderr via bytes.Buffer
  - Cancel ctx after N transcripts
  - Expect: stdout ends with the rendered transcript;
            stderr contains "microphone:", "output capture:", "transcript file:"
  - Acceptance: stdout contains NO "Recording"/TUI chrome (the launch-no-UI contract)
```

This passes today and survives the refactor (it never touches the helper).

### E.2 — TUI controller contract holds across recorder backends — ✅ CAN-WRITE-NOW

`tui/model_test.go` uses a `fakeController` — it is already decoupled from the recorder. The cross-platform invariant: **the TUI_Start/Stop/Pause/Resume/mute/device-select flows do not reference `runtime.GOOS`.** Add a regression comment/test that the model never imports platform-specific audio paths; runnable today by a `grep`-style assertion or by re-running the existing model tests on each OS via CI matrix (DOC-ONLY portion).

### E.3 — TUI launched with no TTY behaves gracefully — ❓ DESIGN-DEPENDENT

**This is an open question for Danny:** when `tui.Run` is invoked and stdin/stdout is not a terminal (e.g. piped, cron, ssh-in-redirected), what is the contract? Options: explicit error to stderr + non-zero exit, or auto-fallback to silent mode. Today `bubbletea` will likely error out poorly. The plan:

```
TestTuiRunFailsCleanlyWithoutTty (DESIGN-DEPENDENT)
  - Given: IO streams that report not-a-tty (e.g. os.Pipe)
  - Expect: tui.Run returns a non-nil error whose message tells the user
            to run in a terminal or pass --silent; the process never hangs.
```

Blocked on what the desired UX is. **Also: the binary must NOT emit any "command line tool" style message** (per the task) — document that as part of this scenario's acceptance text.

---

## F. Launch Behavior & No-TTY Contract — 📋 DOC-ONLY + ❓ DESIGN-DEPENDENT

### F.1 — No-args/help does not print "command line tool" phrasing — 📋 DOC-ONLY (and a cheap filter test)

Pin a contract test in `internal/cli/root_test.go` (table-driven, style of existing `TestSetupLoggingWrites...`):

```
TestRootCommandHelpAvoidsCommandlineToolPhrasing
  - cmd := NewRootCommand(buffer, "v")
  - cmd.SetArgs([]string{"--help"}) and Execute
  - Expect: output contains "transcribe", "Record microphone"
  - MUST NOT contain the literal token "command line tool" (case-insensitive)
```

This is runnable today (the cobra `Short` is already clean) and prevents prose drift.

### F.2 — No-TTY graceful-exit contract — ❓ DESIGN-DEPENDENT (see E.3)

Document the agreed behavior in this doc once Danny decides. Until then: **must not silently exit zero** — at minimum a clear stderr line and exit ≠ 0.

---

## G. Edge Cases — ✅ CAN-WRITE-NOW (mostly)

### G.1 — Missing OPENAI_API_KEY fails before any device work — ✅ CAN-WRITE-NOW

Already proven by `TestPrepareRejectsMissingAPIKeyBeforeDeviceDiscovery`. Add a CLI-level regression that `run()` (root.go) propagates the same error without invoking the recorder. Runnable today.

### G.2 — Missing ffmpeg produces a recorder-validation error, not a crash — ❓ DESIGN-DEPENDENT (today) → ✅ after exec seam

`ExternalRecorder.ffmpeg()` returns an error; `Validate` wraps it. After the helper removal this becomes the **only** failure mode on Windows output too. Assert `Validate(btn)` returns a message containing "ffmpeg" and a `--list-devices` hint. Plan the assertion shape; runnable cross-OS once the exec seam exists (B.1).

### G.3 — No audio devices (empty list) — ✅ CAN-WRITE-NOW

`SelectDevices` with a mic-only or output-less list already returns `"system-output capture device is not selected"` / `"no output capture device"` (see existing `TestSelectDevicesRequiresOutputCapture`, `TestWindowsDShowMicOnlyListDoesNotExposeOutputCapture`). Cross-platform lock: regardless of OS, an output-capture-less device set must fail clearly at selection time, not later during recording.

### G.4 — Interrupted session signal handling — ✅ CAN-WRITE-NOW (session) + 📋 DOC-ONLY (os.Signal)

`root.go` installs `signal.NotifyContext(ctx, os.Interrupt, syscall.SIGTERM)`. The Session layer (`session.Stop`/`Pause` cancellation) is already covered by `TestPauseCancelsActiveRecorderChunks`. Two scoped scenarios:

- ✅ **CAN-WRITE-NOW:** assert that on ctx cancellation, in-flight `RecordChunk` calls for both sources are cancelled and the session `Stop` runs to completion even if a worker is mid-write (extend the blocking-recorder harness already in `app_test.go`).
- 📋 **DOC-ONLY:** manual/CLI-level `Ctrl+C` smoke on each OS confirms a clean `transcript` Markdown file is flushed (since `syscall.SIGTERM` ≠ `os.Interrupt` on Windows, document that only `os.Interrupt`/Ctrl-C is reliable there).

---

## H. Regression — Tests Referencing the Removed Helper — ❓ DESIGN-DEPENDENT (tracked churn)

Per **test-discipline**: in the *same commit* that removes the helper, the following must be updated/removed. Flagged here so nothing lands un-synchronized.

### H.1 — `transcribe/internal/audio/ffmpeg_test.go`

| Existing test | Action when helper removed |
| --- | --- |
| `TestWindowsOutputRecorderUsesPythonWASAPIHelper` | **DELETE** — the entire assertion asserts the Python path exists. |
| `TestWindowsOutputRecorderValidationRequiresHelperScript` | **DELETE/REWRITE** — must instead assert ffmpeg-only validation. |
| `TestWindowsPythonPathPrefersBuildLocalVenv` | **DELETE** — exercises `windowsPythonPath`, which disappears. Replace with `TestFfmpegPathFallsBackToLookup` if a path-resolution seam is added. |
| `TestExternalRecorderRejectsUnresolvedDShowDefaultsBeforeCapture` | **KEEP** — output validation against unresolved dshow defaults stays valid in ffmpeg-only world. |
| `TestWindowsLoopbackOutputArgsUseRawRenderName` | **KEEP** — loopback raw-name contract survives. |

### H.2 — `transcribe/internal/audio/windows_helper.go`

The whole file goes. Its only public effect is on `ExternalRecorder` (via embedded methods `commandForRequest`, `windowsPythonPath`, `windowsHelperScriptPath`, `windowsOutputHelperCommand`). **Ensure no test outside `audio/` references these** — the grep over `transcribe/internal/` today shows references only inside `audio/` and one wiring point in `cli/root.go` (H.4).

### H.3 — `transcribe/internal/audio/ffmpeg.go`

`ExternalRecorder` struct loses `PythonPath`, `HelperScriptPath`. Any test constructing `ExternalRecorder{...PythonPath:...}` (see H.1) must be updated in the same commit. `platform()` may also simplify.

### H.4 — `transcribe/internal/cli/root.go` (the wiring point)

Lines ~128-133 construct `audio.ExternalRecorder{Platform, PythonPath, HelperScriptPath}` reading `TRANSCRIBE_WINDOWS_PYTHON` / `TRANSCRIBE_WINDOWS_AUDIO_HELPER`. After removal this collapses to `audio.ExternalRecorder{FFmpegPath: ...}` (or the renamed type). **No test currently exercises this wiring**, so following test-discipline a new `internal/cli/root_test.go` scenario should be added at refactor time to pin that the recorder is constructed without referencing the removed env vars — this becomes ✅ CAN-WRITE-NOW *after* the API lands.

### H.5 — Other repo files (document for Livingston)

- `build_transcribe_win64.bat` — must drop `.venv` + `audio_capture.py` staging; add a regression note that the WSL build still produces a working self-contained `.exe`. **📋 DOC-ONLY.**
- Root-level `audio_capture.py`, `README.md` Windows-requirements paragraph — to be pruned; not a test, but gated by this same decision.

---

## I. Recommended Test File Layout (ready to scaffold)

| File | Status | Purpose |
| --- | --- | --- |
| `internal/audio/ffmpeg_test.go` (extend) | ✅ CAN-WRITE-NOW | Add `TestInputArgsSelectBackendPerPlatform` table. (After refactor: also `TestWindowsOutputCaptureUsesFfmpegNotPythonHelper`.) |
| `internal/audio/discover_test.go` (extend) | ✅ + ❓ | Add `TestLinuxDiscoveryDegradesGracefullyWhenPactlAbsent`, `TestUnknownOSSynthesizesConfiguredDevicesWithWarning`. Exec-seam variants blocked on Danny's decision. |
| `internal/config/config_test.go` (extend) | ✅ CAN-WRITE-NOW | Add `TestDefaultPathPlatformIndependent` and a no-raw-separator assertion. |
| `internal/app/app_test.go` (extend) | ✅ CAN-WRITE-NOW | Add `TestRunSilentWritesTranscriptToStdoutAndMetricsToStderr`. |
| `internal/cli/root_test.go` (extend) | ✅ CAN-WRITE-NOW | Add `TestRootCommandHelpAvoidsCommandlineToolPhrasing`. |
| `internal/cli/root_test.go` ❓ after refactor | ❓ DESIGN-DEPENDENT | `TestRecorderConstructorIgnoresRemovedHelperEnvVars`. |
| CI / runbook (this doc §A) | 📋 DOC-ONLY | Six-target `CGO_ENABLED=0` build gate. |

---

## J. Open Questions for Danny (blocking the ❓ items)

1. **Recorder API shape:** keep the name `ExternalRecorder` or rename to `FfmpegRecorder`? Does it gain an injectable `Exec`/`FFmpegPath` seam (unblocks B.1/B.3/G.2 cross-OS)?
2. **Windows output capture mechanism in the helper-free design:** IS ffmpeg dshow + a render-endpoint loopback the chosen path everywhere, or does Rusty provide a Go-native WASAPI loopback? (Determines C.2/C.3 exact assertion text.)
3. **No-TTY UX (E.3/F.2):** hard error + non-zero exit, or silent auto-promotion? Required before the binary can be called "terminal-correct".
4. **New/removed CLI flags** (e.g. `--ffmpeg-path`) — drives D.3 and the `RuntimeOverrides` test churn.

---

*End of draft. Reviewer: when Danny's architectural decision lands, convert every ❓ DESIGN-DEPENDENT scenario into a concrete test in this same commit, per test-discipline.*
