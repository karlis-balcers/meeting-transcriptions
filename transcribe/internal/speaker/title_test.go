package speaker

import "testing"

func TestDetectSpeakerFromTitles(t *testing.T) {
	tests := []struct {
		name   string
		titles []string
		want   string
	}{
		{
			name:   "speaking suffix",
			titles: []string{"Microsoft Teams", "Alice Example is speaking - Microsoft Teams"},
			want:   "Alice Example",
		},
		{
			name:   "teams pipe title",
			titles: []string{"Bob B | Microsoft Teams"},
			want:   "Bob B",
		},
		{
			name:   "presenting suffix",
			titles: []string{"Carla is presenting | Microsoft Teams"},
			want:   "Carla",
		},
		{
			name:   "generic only",
			titles: []string{"Microsoft Teams", "Meeting | Microsoft Teams", "Chat | Microsoft Teams"},
			want:   "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := DetectSpeakerFromTitles(tt.titles); got != tt.want {
				t.Fatalf("DetectSpeakerFromTitles() = %q, want %q", got, tt.want)
			}
		})
	}
}
