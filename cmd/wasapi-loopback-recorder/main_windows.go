//go:build windows

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		usageAndExit(2)
	}
	switch os.Args[1] {
	case "list":
		list(os.Args[2:])
	case "record":
		record(os.Args[2:])
	case "--version", "version":
		fmt.Fprintln(os.Stdout, helperVersion)
	default:
		usageAndExit(2)
	}
}

func list(args []string) {
	fs := flag.NewFlagSet("list", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	jsonOutput := fs.Bool("json", false, "emit active render endpoints as JSON")
	if err := fs.Parse(args); err != nil {
		os.Exit(2)
	}
	endpoints, err := listRenderEndpoints()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: list failed: %v\n", helperVersion, err)
		os.Exit(3)
	}
	if *jsonOutput {
		encoder := json.NewEncoder(os.Stdout)
		encoder.SetIndent("", "  ")
		if err := encoder.Encode(endpoints); err != nil {
			fmt.Fprintf(os.Stderr, "%s: write JSON: %v\n", helperVersion, err)
			os.Exit(3)
		}
		return
	}
	for _, endpoint := range endpoints {
		defaultMarker := ""
		if endpoint.Default {
			defaultMarker = "\tdefault"
		}
		fmt.Fprintf(os.Stdout, "%s\t%s%s\n", endpoint.ID, endpoint.Name, defaultMarker)
	}
}

func record(args []string) {
	options, err := parseRecordOptions(args)
	if err != nil {
		fmt.Fprintf(os.Stderr, "%s: %v\n", helperVersion, err)
		os.Exit(2)
	}
	if err := recordLoopback(options); err != nil {
		_ = os.Remove(options.OutputFile)
		fmt.Fprintf(os.Stderr, "%s: record failed for device-id=%q device-name=%q output=%q duration=%s target=%dHz/%dch: %v\n", helperVersion, options.DeviceID, options.DeviceName, options.OutputFile, options.Duration, options.TargetSampleRate, options.TargetChannels, err)
		os.Exit(3)
	}
}

func usageAndExit(code int) {
	fmt.Fprintf(os.Stderr, "Usage:\n  %s record --output-file PATH --duration SECONDS [--device-id ID|default] [--device-name NAME] [--sample-rate 16000] [--channels 1]\n  %s list --json\n", os.Args[0], os.Args[0])
	os.Exit(code)
}
