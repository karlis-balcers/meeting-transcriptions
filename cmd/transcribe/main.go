package main

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/cli"
)

var version = "dev"

func main() {
	// Ensure a console is attached before the Bubble Tea TUI / Cobra CLI runs:
	// a double-clicked or shell-extension-launched console-subsystem exe may
	// start without one. No-op on non-Windows.
	attachParentConsole()

	// Disable Cobra's mousetrap: when transcribe.exe is launched by double-
	// clicking in Explorer, cobra (via inconshreveable/mousetrap) detects the
	// parent process as explorer.exe, prints "This is a command line tool /
	// You need to open cmd.exe...", and os.Exit(1) before the TUI ever runs.
	// Blanking MousetrapHelpText suppresses that early exit so the double-click
	// UX works (the attachParentConsole() above then provides the TTY). This
	// assignment is safe cross-platform: MousetrapHelpText is a plain exported
	// var in cobra.go; only the Windows-only check in command_win.go reads it.
	cobra.MousetrapHelpText = ""

	cmd := cli.NewRootCommand(cli.IO{
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}, version)
	if err := cmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
