package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDetermineCategory(t *testing.T) {
	cases := []struct {
		name     string
		explicit string
		expected string
	}{
		{"model.safetensors", "", "checkpoint"},
		{"model.gguf", "", "gguf"},
		{"cool_lora.pt", "", "lora"},
		{"custom.bin", "vae", "vae"},
		{"unknown.xyz", "", "checkpoint"},
	}
	for _, tc := range cases {
		got := determineCategory(tc.name, tc.explicit)
		if got != tc.expected {
			t.Fatalf("determineCategory(%q, %q)=%q want %q", tc.name, tc.explicit, got, tc.expected)
		}
	}
}

func TestLoadMetadataJSON(t *testing.T) {
	content := `{"models": [{"filename": "a.safetensors", "category": "lora", "huggingface": {"repo_id": "org/model"}}]}`
	tmp := filepath.Join(t.TempDir(), "models.json")
	if err := os.WriteFile(tmp, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	meta, err := loadMetadataFile(tmp)
	if err != nil {
		t.Fatalf("loadMetadataFile failed: %v", err)
	}
	entry, ok := meta["a.safetensors"]
	if !ok {
		t.Fatalf("metadata entry missing")
	}
	if entry.Category != "lora" {
		t.Fatalf("expected category 'lora', got %q", entry.Category)
	}
	if entry.HuggingFace["repo_id"] != "org/model" {
		t.Fatalf("expected repo_id to be preserved")
	}
}

func TestLoadMetadataYAML(t *testing.T) {
	content := `models:
  - filename: sample.safetensors
    category: checkpoint
    huggingface:
      repo_id: sample/repo
      revision: main
    civitai:
      model_id: "123"
      version_id: "456"
`
	tmp := filepath.Join(t.TempDir(), "models.yaml")
	if err := os.WriteFile(tmp, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	meta, err := loadMetadataFile(tmp)
	if err != nil {
		t.Fatalf("loadMetadataFile failed: %v", err)
	}
	entry := meta["sample.safetensors"]
	if entry.HuggingFace["repo_id"] != "sample/repo" {
		t.Fatalf("expected huggingface repo_id")
	}
	if entry.CivitAI["version_id"] != "456" {
		t.Fatalf("expected civitai version id")
	}
}

func TestOrganizerRunCopiesAndWrites(t *testing.T) {
	temp := t.TempDir()
	source := filepath.Join(temp, "source")
	dest := filepath.Join(temp, "dest")
	if err := os.MkdirAll(source, 0o755); err != nil {
		t.Fatalf("mkdir source: %v", err)
	}

	filePath := filepath.Join(source, "model.safetensors")
	if err := os.WriteFile(filePath, []byte("dummy-weights"), 0o644); err != nil {
		t.Fatalf("write source file: %v", err)
	}

	meta := metadataIndex{
		"model.safetensors": {
			Category: "lora",
			HuggingFace: map[string]string{
				"repo_id":  "org/model",
				"filename": "model.safetensors",
			},
			CivitAI: map[string]string{
				"model_id":   "1",
				"version_id": "2",
			},
			Extra: map[string]string{
				"note": "test",
			},
		},
	}

	organizer := newOrganizer(source, dest, false, meta)
	organizer.FetchHuggingFace = func(meta SourceMetadata, token string) map[string]interface{} {
		return map[string]interface{}{
			"card_data": map[string]interface{}{"recommended_batch": "1"},
		}
	}
	organizer.FetchCivitAI = func(meta SourceMetadata, token string) map[string]interface{} {
		return map[string]interface{}{
			"preview_images": []map[string]interface{}{{"url": "https://example.com/image.png"}},
			"hashes":         map[string]string{"sha256": "remote"},
		}
	}

	manifest, err := organizer.run()
	if err != nil {
		t.Fatalf("organizer run failed: %v", err)
	}

	expectedDest := filepath.Join(dest, "models", "loras", "model.safetensors")
	if _, err := os.Stat(expectedDest); err != nil {
		t.Fatalf("destination file missing: %v", err)
	}

	jsonPath := expectedDest + ".metadata.json"
	data, err := os.ReadFile(jsonPath)
	if err != nil {
		t.Fatalf("metadata json missing: %v", err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("metadata json invalid: %v", err)
	}
	if parsed["category"].(string) != "lora" {
		t.Fatalf("unexpected category in metadata json")
	}

	yamlPath := expectedDest + ".metadata.yaml"
	yamlContent, err := os.ReadFile(yamlPath)
	if err != nil {
		t.Fatalf("metadata yaml missing: %v", err)
	}
	if !strings.Contains(string(yamlContent), "preview_images") {
		t.Fatalf("yaml metadata missing civitai previews: %s", string(yamlContent))
	}

	manifestJSON := filepath.Join(dest, "models_manifest.json")
	if _, err := os.Stat(manifestJSON); err != nil {
		t.Fatalf("manifest json missing: %v", err)
	}
	if len(manifest.Models) != 1 {
		t.Fatalf("expected single model in manifest")
	}
	if manifest.Models[0].Category != "lora" {
		t.Fatalf("manifest category mismatch")
	}
}
