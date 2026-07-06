//go:build windows

package main

import (
	"os"
	"syscall"
)

// attachParentConsole ensures the process has a usable console when launched
// detached (e.g. double-clicked from Explorer, a shell extension, or Task
// Scheduler). transcribe is a Bubble Tea terminal TUI that needs a real TTY;
// we deliberately keep the CONSOLE subsystem (no -H windowsgui). When this
// process already owns a console the call is a no-op.
//
// Strategy: if there is no console window for this process, call
// AttachConsole(ATTACH_PARENT_PROCESS) and reopen os.Stdout / os.Stderr /
// os.Stdin against the attached console's handles (CONOUT$ / CONIN$). All
// failures are swallowed (best effort): a missing console is logged to a
// sidecar file instead of crashing, so Task Scheduler launches still degrade
// gracefully.
func attachParentConsole() {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")
	getConsoleWindow := kernel32.NewProc("GetConsoleWindow")
	attachConsole := kernel32.NewProc("AttachConsole")

	// GetConsoleWindow returns the HWND of the current console, or 0 if none.
	// A non-zero HWND means we already own or are attached to a console.
	if hwnd, _, _ := getConsoleWindow.Call(); hwnd != 0 {
		return
	}

	const attachParentProcess = ^uintptr(0) // ATTACH_PARENT_PROCESS == -1 == ^uintptr(0)
	if r, _, _ := attachConsole.Call(attachParentProcess); r == 0 {
		// AttachConsole failed: the parent has no console to attach to (e.g.
		// launched from Task Scheduler / service). Fail gracefully.
		writeConsoleAttachFailure("AttachConsole returned 0; parent has no console")
		return
	}

	if h, err := syscall.CreateFile(
		syscall.StringToUTF16Ptr("CONOUT$"),
		syscall.GENERIC_WRITE,
		syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_READ,
		nil,
		syscall.OPEN_EXISTING,
		0,
		0,
	); err == nil {
		os.Stdout = os.NewFile(uintptr(h), "CONOUT$")
		os.Stderr = os.NewFile(uintptr(h), "CONOUT$")
	}
	if h, err := syscall.CreateFile(
		syscall.StringToUTF16Ptr("CONIN$"),
		syscall.GENERIC_READ,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE,
		nil,
		syscall.OPEN_EXISTING,
		0,
		0,
	); err == nil {
		os.Stdin = os.NewFile(uintptr(h), "CONIN$")
	}
}

// writeConsoleAttachFailure records a failure to a sidecar file so a detached
// process (Task Scheduler / service) leaves a breadcrumb without crashing the
// TUI.
func writeConsoleAttachFailure(msg string) {
	const fallbackPath = "transcribe-console.log"
	f, err := os.OpenFile(fallbackPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		return
	}
	defer f.Close()
	_, _ = f.WriteString(msg + "\n")
}
