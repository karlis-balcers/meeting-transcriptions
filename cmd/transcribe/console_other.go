//go:build !windows

package main

// attachParentConsole is a no-op on non-Windows: POSIX shells always attach a
// controlling terminal to the process, and the Bubble Tea TUI relies on it.
// Keeping a stub keeps main.go build-tag-free.
func attachParentConsole() {}
