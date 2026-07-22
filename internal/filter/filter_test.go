package filter

import "testing"

func TestFilterDropsArtifactsButKeepsKeywords(t *testing.T) {
	f, warnings := New(Config{
		MinChars: 3,
		Exact:    []string{"LAMPA"},
		Prefixes: []string{"[Music]"},
		Contains: []string{"thank you for watching"},
		Regex:    []string{`^\(.*\)$`, "["},
		Keywords: []string{"AI"},
	})
	if len(warnings) != 1 {
		t.Fatalf("expected invalid regex warning, got %v", warnings)
	}
	cases := map[string]bool{
		"LAMPA":                     false,
		"[Music]":                   false,
		"Thank you for watching us": false,
		"(noise)":                   false,
		"ok":                        false,
		"AI":                        true,
		"Useful meeting note":       true,
	}
	for input, want := range cases {
		if got := f.Keep(input); got != want {
			t.Fatalf("Keep(%q)=%v want %v", input, got, want)
		}
	}
}
