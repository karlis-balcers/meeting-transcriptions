package main

import (
	"fmt"
	"os"

	"github.com/karlis-balcers/meeting-transcriptions/transcribe/internal/cli"
)

var version = "dev"

func main() {
	cmd := cli.NewRootCommand(cli.IO{
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}, version)
	if err := cmd.Execute(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
