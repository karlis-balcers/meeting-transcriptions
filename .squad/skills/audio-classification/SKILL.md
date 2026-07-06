# Skill: Audio device classification (display vs runtime-capable)

> When an audio device is surfaced for the user to SEE but cannot actually be
> OPENED by the capture backend, keep the two concerns strictly separated.
> Display devices stay in the picker; runtime devices are the only ones the
> recorder ever tries to open.

## When to apply

- You're adding or refining an audio discovery path that synthesizes devices from a non-ffmpeg source (Windows `Get-PnpDevice -Class AudioEndpoint`, a future macOS CoreAudio enumeration, etc.).
- You're adding a new capture backend or a new device kind to an existing backend.
- You're debugging a "falling back → failed → falling back" cascading loop in the runtime capture path.
- You're wiring the runtime fallback chain (`audio.OutputCaptureCandidates`, `app/session.go::nextOutputCaptureCandidate`).

## The core rule

A device has **three** distinct identities, and confusing them is the bug:

1. **ID** — the string the capture backend actually opens (e.g. the ffmpeg DirectShow friendly name, the Pulse source id). This is what `audio=<ID>` / `-i <ID>` receives.
2. **Display name** — what the user sees in `--list-devices` / the picker. This may be prettified for the user and may contain tokens that look like capabilities.
3. **Aliases** — secondary identifiers (instance IDs, numeric indices, alternative names) that ALSO prove the device is a real, openable source.

**Never classify a device's capability by inspecting a display name you generate yourself.** If discovery formats a synthesized device as `"<render> [Loopback]"`, then a classifier reading the display name will see the literal token "loopback" and wrongly conclude the device is a real loopback source. Classify capability by inspecting the **ID** and **aliases** — those carry the real backend signature.

## Canonical Go implementation (this repo)

`transcribe/internal/audio/audio.go`:

- `SynthesizedDirectShowRenderEndpoint(device Device) bool` — true iff `Source == SourceOutput && Backend == "dshow"` and NEITHER the ID nor any alias carries a real loopback signature (`likelyDShowLoopback`). Inspects ID + aliases only, NOT `device.Name`.
- `runtimeCapableOutputDevice(device Device) bool` — non-dshow backends always true; dshow true iff NOT synthesized.
- `OutputCaptureCandidates(devices, current)` — filters the candidate pool through `runtimeCapableOutputDevice`. `current`/self may surface at index 0 (so the failure message still names the attempted device and the fallback cursor can advance via `[1]`), but every actual fallback TARGET `candidates[1:]` is real.
- `OutputCaptureUnavailableGuidance(devices) string` — returns `windowsOutputCaptureWarning` when output devices exist but none are runtime-capable; empty otherwise. Inspects device metadata (not `runtime.GOOS`), so it's unit-testable on Linux/CI.

`transcribe/internal/audio/discover.go`:

- `devicesFromWindowsAudioEndpoints` is where the poisoned display name is born: it sets `Device.Name = windowsLoopbackDisplayName(raw)` i.e. `"<render> [Loopback]"`. This is correct and SHOULD stay — it tells the user "this is a render endpoint we'd loopback IF we could". The poisoned name is fine for DISPLAY; it's only a problem if a capability classifier reads it.

## Anti-patterns to reject in review

- A device whose `Name` contains `loopback` / `monitor` / `Stereo Mix` being treated as runtime-capable. Read the ID/aliases, not the display name.
- The runtime fallback chain (`OutputCaptureCandidates`) returning synthesized render endpoints. The chain must only ever contain devices ffmpeg can actually open.
- A bare `output capture failed for X: ffmpeg failed…` error when NO runtime-capable output device exists. Always pair the error with the actionable remediation (`OutputCaptureUnavailableGuidance`) naming Stereo Mix / VB-CABLE / Voicemeeter / `--list-devices`.
- Stripping synthesized render endpoints out of `--list-devices` or the picker. Display discovery must keep surfacing them so users can see what render endpoints exist; only the runtime path excludes them.

## Test patterns to add alongside any new device kind

- The classifier returns the expected verdict on a hand-built `Device` literal mirroring exactly what the discovery function emits (same ID, same `[Loopback]` display name, same alias shape).
- `OutputCaptureCandidates` excludes synthesized devices from fallback TARGETS (`candidates[1:]`) but keeps real loopback/monitor sources.
- When only synthesized devices remain, the fallback chain collapses to `len==1` (self only) instead of cycling — this is the regression that stops cascading "falling back → failed" churn.
- `OutputCaptureUnavailableGuidance` fires for synthesized-only sets and stays silent when a real source (or no output at all) is present.
- Non-target backends (pulse `.monitor`, avfoundation, test `fake` backends) pass through unchanged.

## Reference

- Enforcing decision: `.squad/decisions/inbox/livingston-synthesized-render-endpoints-display-only.md`
- Verified memory: `/memories/repo/audio-devices.md` (2026-05-29 and 2026-07-03 entries, plus the `windowsOutputCaptureWarning` constant in `internal/audio/discover.go`).
- Parent ADR: the 2026-07-03 Path-C decision in `.squad/decisions.md` (standalone terminal binary, ffmpeg-unified).
