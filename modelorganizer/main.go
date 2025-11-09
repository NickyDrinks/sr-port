package main

import (
	"bufio"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Mapping between semantic categories and the default relative directories many frontends expect.
var defaultSubdirs = map[string]string{
	"checkpoint":        "models/checkpoints",
	"diffusion":         "models/checkpoints",
	"lora":              "models/loras",
	"lycoris":           "models/loras",
	"textual_inversion": "models/embeddings",
	"embedding":         "models/embeddings",
	"clip":              "models/clip",
	"vae":               "models/vae",
	"upscale":           "models/upscale_models",
	"controlnet":        "models/controlnet",
	"ipadapter":         "models/ipadapter",
	"gaussian_splat":    "models/gaussian_splatting",
	"dynamo":            "models/custom",
	"gguf":              "models/gguf",
}

var extensionHints = map[string]string{
	".safetensors": "checkpoint",
	".ckpt":        "checkpoint",
	".pt":          "checkpoint",
	".bin":         "checkpoint",
	".gguf":        "gguf",
	".ggml":        "gguf",
	".pth":         "checkpoint",
	".onnx":        "custom",
}

var nameHints = map[string]string{
	"lora":       "lora",
	"lycoris":    "lycoris",
	"controlnet": "controlnet",
	"embedding":  "embedding",
	"textinv":    "embedding",
	"vae":        "vae",
	"clip":       "clip",
	"ipadapter":  "ipadapter",
	"upscale":    "upscale",
}

// SourceMetadata describes the hints supplied for a particular model file.
type SourceMetadata struct {
	Category    string
	HuggingFace map[string]string
	CivitAI     map[string]string
	Extra       map[string]string
}

type metadataIndex map[string]SourceMetadata

type hashes struct {
	SHA256 string `json:"sha256"`
	SHA512 string `json:"sha512"`
}

// ModelRecord captures the resulting organisation data for one model file.
type ModelRecord struct {
	Category     string                 `json:"category"`
	Destination  string                 `json:"destination"`
	Hashes       hashes                 `json:"hashes"`
	Size         int64                  `json:"size"`
	MetadataJSON string                 `json:"metadata_json"`
	MetadataYAML string                 `json:"metadata_yaml"`
	Source       map[string]interface{} `json:"source"`
}

// Manifest summarises all organised models in the destination directory.
type Manifest struct {
	Root   string        `json:"root"`
	Models []ModelRecord `json:"models"`
}

// Organizer performs the filesystem work.
type Organizer struct {
	SourceDir        string
	DestinationDir   string
	DryRun           bool
	HuggingFaceToken string
	CivitAIToken     string
	Metadata         metadataIndex

	FetchHuggingFace func(SourceMetadata, string) map[string]interface{}
	FetchCivitAI     func(SourceMetadata, string) map[string]interface{}
}

func (o *Organizer) run() (Manifest, error) {
	if o.SourceDir == "" {
		return Manifest{}, errors.New("source directory is required")
	}
	if o.DestinationDir == "" {
		return Manifest{}, errors.New("destination directory is required")
	}

	info, err := os.Stat(o.SourceDir)
	if err != nil {
		return Manifest{}, fmt.Errorf("failed to stat source directory: %w", err)
	}
	if !info.IsDir() {
		return Manifest{}, fmt.Errorf("source path %q is not a directory", o.SourceDir)
	}

	if !o.DryRun {
		if err := os.MkdirAll(o.DestinationDir, 0o755); err != nil {
			return Manifest{}, fmt.Errorf("unable to create destination: %w", err)
		}
	}

	entries, err := os.ReadDir(o.SourceDir)
	if err != nil {
		return Manifest{}, fmt.Errorf("failed to list source directory: %w", err)
	}

	manifest := Manifest{Root: o.DestinationDir}

	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		sourcePath := filepath.Join(o.SourceDir, name)
		meta := o.Metadata[name]
		category := determineCategory(name, meta.Category)
		destDir := resolveDestination(o.DestinationDir, category)
		destPath := filepath.Join(destDir, name)

		if !o.DryRun {
			if err := os.MkdirAll(destDir, 0o755); err != nil {
				return Manifest{}, fmt.Errorf("create destination folder: %w", err)
			}
		}

		fileHashes, size, err := copyAndHash(sourcePath, destPath, o.DryRun)
		if err != nil {
			return Manifest{}, fmt.Errorf("copy %s: %w", name, err)
		}

		sourceInfo := map[string]interface{}{
			"original_path": sourcePath,
		}
		if len(meta.HuggingFace) > 0 && o.FetchHuggingFace != nil {
			sourceInfo["huggingface"] = mergeMaps(meta.HuggingFace, o.FetchHuggingFace(meta, o.HuggingFaceToken))
		} else if len(meta.HuggingFace) > 0 {
			sourceInfo["huggingface"] = cloneStringMap(meta.HuggingFace)
		}
		if len(meta.CivitAI) > 0 && o.FetchCivitAI != nil {
			sourceInfo["civitai"] = mergeMaps(meta.CivitAI, o.FetchCivitAI(meta, o.CivitAIToken))
		} else if len(meta.CivitAI) > 0 {
			sourceInfo["civitai"] = cloneStringMap(meta.CivitAI)
		}
		if len(meta.Extra) > 0 {
			for k, v := range meta.Extra {
				sourceInfo[k] = v
			}
		}

		jsonPath := destPath + ".metadata.json"
		yamlPath := destPath + ".metadata.yaml"
		record := ModelRecord{
			Category:     category,
			Destination:  destPath,
			Hashes:       fileHashes,
			Size:         size,
			MetadataJSON: jsonPath,
			MetadataYAML: yamlPath,
			Source:       sourceInfo,
		}

		if !o.DryRun {
			payload := map[string]interface{}{
				"category": category,
				"hashes": map[string]string{
					"sha256": fileHashes.SHA256,
					"sha512": fileHashes.SHA512,
				},
				"size":   size,
				"source": sourceInfo,
			}
			if err := writeJSON(jsonPath, payload); err != nil {
				return Manifest{}, fmt.Errorf("write json metadata: %w", err)
			}
			if err := writeYAML(yamlPath, payload); err != nil {
				return Manifest{}, fmt.Errorf("write yaml metadata: %w", err)
			}
		}

		manifest.Models = append(manifest.Models, record)
	}

	if !o.DryRun && len(manifest.Models) > 0 {
		if err := writeJSON(filepath.Join(o.DestinationDir, "models_manifest.json"), manifestToMap(manifest)); err != nil {
			return Manifest{}, fmt.Errorf("write manifest json: %w", err)
		}
		if err := writeYAML(filepath.Join(o.DestinationDir, "models_manifest.yaml"), manifestToMap(manifest)); err != nil {
			return Manifest{}, fmt.Errorf("write manifest yaml: %w", err)
		}
	}

	return manifest, nil
}

func determineCategory(filename string, explicit string) string {
	if explicit != "" {
		category := strings.ToLower(explicit)
		if _, ok := defaultSubdirs[category]; ok {
			return category
		}
	}
	lower := strings.ToLower(strings.TrimSuffix(filename, filepath.Ext(filename)))
	for hint, category := range nameHints {
		if strings.Contains(lower, hint) {
			return category
		}
	}
	category, ok := extensionHints[strings.ToLower(filepath.Ext(filename))]
	if !ok {
		category = "checkpoint"
	}
	if _, ok := defaultSubdirs[category]; !ok {
		return "custom"
	}
	return category
}

func resolveDestination(base, category string) string {
	if subdir, ok := defaultSubdirs[category]; ok {
		return filepath.Join(base, subdir)
	}
	return filepath.Join(base, filepath.Join("models", category))
}

func copyAndHash(source, destination string, dryRun bool) (hashes, int64, error) {
	file, err := os.Open(source)
	if err != nil {
		return hashes{}, 0, err
	}
	defer file.Close()

	var writers []io.Writer
	sha := sha256.New()
	sha512h := sha512.New()
	writers = append(writers, sha, sha512h)

	var destFile *os.File
	if !dryRun {
		destFile, err = os.OpenFile(destination, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o644)
		if err != nil {
			return hashes{}, 0, err
		}
		defer destFile.Close()
		writers = append(writers, destFile)
	}

	written, err := io.Copy(io.MultiWriter(writers...), file)
	if err != nil {
		return hashes{}, 0, err
	}

	if !dryRun {
		if err := os.Chtimes(destination, time.Now(), time.Now()); err != nil {
			return hashes{}, 0, err
		}
		if err := os.Chmod(destination, 0o644); err != nil {
			return hashes{}, 0, err
		}
	}

	return hashes{SHA256: fmt.Sprintf("%x", sha.Sum(nil)), SHA512: fmt.Sprintf("%x", sha512h.Sum(nil))}, written, nil
}

func mergeMaps(base map[string]string, extra map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for k, v := range base {
		result[k] = v
	}
	for k, v := range extra {
		result[k] = v
	}
	return result
}

func cloneStringMap(src map[string]string) map[string]string {
	dup := make(map[string]string, len(src))
	for k, v := range src {
		dup[k] = v
	}
	return dup
}

func writeJSON(path string, data interface{}) error {
	payload, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}
	payload = append(payload, '\n')
	return os.WriteFile(path, payload, 0o644)
}

func writeYAML(path string, data interface{}) error {
	var builder strings.Builder
	emitYAML(&builder, normalizeYAMLValue(data), 0)
	builder.WriteByte('\n')
	return os.WriteFile(path, []byte(builder.String()), 0o644)
}

func normalizeYAMLValue(value interface{}) interface{} {
	switch v := value.(type) {
	case Manifest:
		return manifestToMap(v)
	case map[string]interface{}, []interface{}, map[string]string:
		return v
	case []ModelRecord:
		out := make([]interface{}, 0, len(v))
		for _, m := range v {
			out = append(out, modelRecordToMap(m))
		}
		return out
	case ModelRecord:
		return modelRecordToMap(v)
	default:
		return v
	}
}

func emitYAML(builder *strings.Builder, data interface{}, indent int) {
	padding := strings.Repeat(" ", indent)
	switch value := data.(type) {
	case map[string]interface{}:
		keys := make([]string, 0, len(value))
		for k := range value {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, key := range keys {
			builder.WriteString(padding)
			builder.WriteString(key)
			builder.WriteString(":")
			switch child := value[key].(type) {
			case map[string]interface{}, []interface{}:
				builder.WriteByte('\n')
				emitYAML(builder, child, indent+2)
			default:
				builder.WriteByte(' ')
				builder.WriteString(formatScalar(child))
				builder.WriteByte('\n')
			}
		}
	case map[string]string:
		keys := make([]string, 0, len(value))
		for k := range value {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, key := range keys {
			builder.WriteString(padding)
			builder.WriteString(key)
			builder.WriteString(": ")
			builder.WriteString(formatScalar(value[key]))
			builder.WriteByte('\n')
		}
	case []interface{}:
		for _, item := range value {
			builder.WriteString(padding)
			builder.WriteString("-")
			switch child := item.(type) {
			case map[string]interface{}, []interface{}:
				builder.WriteByte('\n')
				emitYAML(builder, child, indent+2)
			default:
				builder.WriteByte(' ')
				builder.WriteString(formatScalar(child))
				builder.WriteByte('\n')
			}
		}
	default:
		builder.WriteString(padding)
		builder.WriteString(formatScalar(value))
		builder.WriteByte('\n')
	}
}

func manifestToMap(manifest Manifest) map[string]interface{} {
	models := make([]interface{}, 0, len(manifest.Models))
	for _, model := range manifest.Models {
		models = append(models, modelRecordToMap(model))
	}
	return map[string]interface{}{
		"root":   manifest.Root,
		"models": models,
	}
}

func modelRecordToMap(record ModelRecord) map[string]interface{} {
	hashesMap := map[string]string{
		"sha256": record.Hashes.SHA256,
		"sha512": record.Hashes.SHA512,
	}
	return map[string]interface{}{
		"category":      record.Category,
		"destination":   record.Destination,
		"hashes":        hashesMap,
		"size":          record.Size,
		"metadata_json": record.MetadataJSON,
		"metadata_yaml": record.MetadataYAML,
		"source":        record.Source,
	}
}

func formatScalar(value interface{}) string {
	switch v := value.(type) {
	case nil:
		return "null"
	case string:
		if strings.ContainsAny(v, "#:{}[]\"\n\r\t") || strings.HasPrefix(v, " ") || strings.HasSuffix(v, " ") {
			b, _ := json.Marshal(v)
			return string(b)
		}
		return v
	case fmt.Stringer:
		return formatScalar(v.String())
	case bool:
		if v {
			return "true"
		}
		return "false"
	case int, int8, int16, int32, int64:
		return fmt.Sprintf("%d", v)
	case uint, uint8, uint16, uint32, uint64:
		return fmt.Sprintf("%d", v)
	case float32, float64:
		return fmt.Sprintf("%v", v)
	default:
		b, _ := json.Marshal(v)
		return string(b)
	}
}

func loadMetadataFile(path string) (metadataIndex, error) {
	content, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	trimmed := strings.TrimSpace(string(content))
	if trimmed == "" {
		return metadataIndex{}, nil
	}
	if trimmed[0] == '{' {
		return parseMetadataJSON(trimmed)
	}
	return parseMetadataYAML(content)
}

type rawMetadata struct {
	Models []map[string]interface{} `json:"models"`
}

func parseMetadataJSON(raw string) (metadataIndex, error) {
	var decoded rawMetadata
	if err := json.Unmarshal([]byte(raw), &decoded); err != nil {
		return nil, err
	}
	return buildMetadataIndex(decoded.Models)
}

func parseMetadataYAML(content []byte) (metadataIndex, error) {
	scanner := bufio.NewScanner(strings.NewReader(string(content)))
	scanner.Split(bufio.ScanLines)

	var entries []map[string]interface{}
	current := make(map[string]interface{})
	context := ""

	for scanner.Scan() {
		line := scanner.Text()
		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}
		if trimmed == "models:" {
			continue
		}
		if strings.HasPrefix(trimmed, "- ") {
			if len(current) > 0 {
				entries = append(entries, current)
			}
			current = make(map[string]interface{})
			context = ""
			kv := strings.TrimSpace(strings.TrimPrefix(trimmed, "- "))
			if kv != "" {
				key, value := splitKeyValue(kv)
				if key != "" {
					current[key] = value
				}
			}
			continue
		}
		if strings.HasSuffix(trimmed, ":") {
			context = strings.TrimSuffix(trimmed, ":")
			continue
		}
		key, value := splitKeyValue(trimmed)
		if key == "" {
			continue
		}
		switch context {
		case "":
			current[key] = value
		case "huggingface", "hugging_face":
			ensureNestedMap(current, "huggingface")[key] = fmt.Sprintf("%v", value)
		case "civitai":
			ensureNestedMap(current, "civitai")[key] = fmt.Sprintf("%v", value)
		default:
			ensureNestedMap(current, context)[key] = fmt.Sprintf("%v", value)
		}
	}
	if len(current) > 0 {
		entries = append(entries, current)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return buildMetadataIndex(entries)
}

func splitKeyValue(line string) (string, interface{}) {
	parts := strings.SplitN(line, ":", 2)
	if len(parts) != 2 {
		return "", nil
	}
	key := strings.TrimSpace(parts[0])
	value := strings.TrimSpace(parts[1])
	if value == "" {
		return key, ""
	}
	if strings.HasPrefix(value, "\"") && strings.HasSuffix(value, "\"") && len(value) >= 2 {
		unquoted, err := strconv.Unquote(value)
		if err == nil {
			return key, unquoted
		}
	}
	if i, err := strconv.Atoi(value); err == nil {
		return key, i
	}
	if strings.EqualFold(value, "true") || strings.EqualFold(value, "false") {
		return key, strings.EqualFold(value, "true")
	}
	return key, value
}

func ensureNestedMap(entry map[string]interface{}, key string) map[string]string {
	if existing, ok := entry[key]; ok {
		if cast, ok := existing.(map[string]string); ok {
			return cast
		}
		if cast, ok := existing.(map[string]interface{}); ok {
			converted := make(map[string]string)
			for k, v := range cast {
				converted[k] = fmt.Sprintf("%v", v)
			}
			entry[key] = converted
			return converted
		}
	}
	m := make(map[string]string)
	entry[key] = m
	return m
}

func buildMetadataIndex(entries []map[string]interface{}) (metadataIndex, error) {
	index := make(metadataIndex)
	for _, entry := range entries {
		rawFilename, ok := entry["filename"]
		if !ok {
			return nil, errors.New("metadata entry is missing filename")
		}
		filename := fmt.Sprintf("%v", rawFilename)
		meta := SourceMetadata{
			HuggingFace: make(map[string]string),
			CivitAI:     make(map[string]string),
			Extra:       make(map[string]string),
		}
		if category, ok := entry["category"]; ok {
			meta.Category = fmt.Sprintf("%v", category)
		}
		if hfRaw, ok := entry["huggingface"]; ok {
			switch val := hfRaw.(type) {
			case map[string]string:
				for k, v := range val {
					meta.HuggingFace[k] = v
				}
			case map[string]interface{}:
				for k, v := range val {
					meta.HuggingFace[k] = fmt.Sprintf("%v", v)
				}
			}
		}
		if cvRaw, ok := entry["civitai"]; ok {
			switch val := cvRaw.(type) {
			case map[string]string:
				for k, v := range val {
					meta.CivitAI[k] = v
				}
			case map[string]interface{}:
				for k, v := range val {
					meta.CivitAI[k] = fmt.Sprintf("%v", v)
				}
			}
		}
		for key, value := range entry {
			if key == "filename" || key == "category" || key == "huggingface" || key == "civitai" {
				continue
			}
			meta.Extra[key] = fmt.Sprintf("%v", value)
		}
		index[filename] = meta
	}
	return index, nil
}

func defaultHuggingFaceFetcher(meta SourceMetadata, token string) map[string]interface{} {
	repoID := meta.HuggingFace["repo_id"]
	if repoID == "" {
		return map[string]interface{}{}
	}
	url := fmt.Sprintf("https://huggingface.co/api/models/%s", repoID)
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	if token == "" {
		token = os.Getenv("HF_API_TOKEN")
	}
	if token != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	}
	query := req.URL.Query()
	if revision := meta.HuggingFace["revision"]; revision != "" {
		query.Set("revision", revision)
	}
	req.URL.RawQuery = query.Encode()

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer resp.Body.Close()

	var decoded map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return map[string]interface{}{"error": err.Error()}
	}

	var fileInfo map[string]interface{}
	targetFilename := meta.HuggingFace["filename"]
	if siblings, ok := decoded["siblings"].([]interface{}); ok && targetFilename != "" {
		for _, sibling := range siblings {
			siblingMap, ok := sibling.(map[string]interface{})
			if !ok {
				continue
			}
			if fmt.Sprintf("%v", siblingMap["rfilename"]) == targetFilename {
				fileInfo = siblingMap
				break
			}
		}
	}
	card := map[string]interface{}{}
	if cd, ok := decoded["cardData"].(map[string]interface{}); ok {
		card = cd
	}

	result := map[string]interface{}{
		"api": decoded,
	}
	if len(fileInfo) > 0 {
		result["file"] = fileInfo
	}
	if len(card) > 0 {
		result["card_data"] = card
	}
	return result
}

func defaultCivitAIFetcher(meta SourceMetadata, token string) map[string]interface{} {
	versionID := meta.CivitAI["version_id"]
	modelID := meta.CivitAI["model_id"]
	if versionID == "" && modelID == "" {
		return map[string]interface{}{}
	}
	endpoint := ""
	if versionID != "" {
		endpoint = fmt.Sprintf("https://civitai.com/api/v1/model-versions/%s", versionID)
	} else {
		endpoint = fmt.Sprintf("https://civitai.com/api/v1/models/%s", modelID)
	}
	req, err := http.NewRequest(http.MethodGet, endpoint, nil)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	if token == "" {
		token = os.Getenv("CIVITAI_API_TOKEN")
	}
	if token != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token))
	}

	client := &http.Client{Timeout: 15 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return map[string]interface{}{"error": err.Error()}
	}
	defer resp.Body.Close()

	var decoded map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return map[string]interface{}{"error": err.Error()}
	}

	result := map[string]interface{}{"api": decoded}
	if images, ok := decoded["images"].([]interface{}); ok {
		previews := make([]map[string]interface{}, 0, len(images))
		for _, img := range images {
			if imap, ok := img.(map[string]interface{}); ok {
				preview := map[string]interface{}{}
				if url, ok := imap["url"].(string); ok {
					preview["url"] = url
				}
				if meta, ok := imap["meta"].(map[string]interface{}); ok {
					preview["metadata"] = meta
				}
				previews = append(previews, preview)
			}
		}
		if len(previews) > 0 {
			result["preview_images"] = previews
		}
	}
	if files, ok := decoded["files"].([]interface{}); ok {
		hashes := make(map[string]string)
		for _, fileEntry := range files {
			if fmap, ok := fileEntry.(map[string]interface{}); ok {
				if fh, ok := fmap["hashes"].(map[string]interface{}); ok {
					for k, v := range fh {
						hashes[fmt.Sprintf("%v", k)] = fmt.Sprintf("%v", v)
					}
				}
			}
		}
		if len(hashes) > 0 {
			result["hashes"] = hashes
		}
	}
	return result
}

func newOrganizer(source, destination string, dryRun bool, metadata metadataIndex) *Organizer {
	return &Organizer{
		SourceDir:        source,
		DestinationDir:   destination,
		DryRun:           dryRun,
		Metadata:         metadata,
		FetchHuggingFace: defaultHuggingFaceFetcher,
		FetchCivitAI:     defaultCivitAIFetcher,
	}
}

func main() {
	var sourceDir string
	var destinationDir string
	var metadataPath string
	var dryRun bool
	var hfToken string
	var civitaiToken string

	flag.StringVar(&sourceDir, "source", "", "Directory containing model files to organise")
	flag.StringVar(&destinationDir, "destination", "", "output root directory")
	flag.StringVar(&metadataPath, "metadata", "", "Optional JSON or YAML file describing model metadata")
	flag.BoolVar(&dryRun, "dry-run", false, "Scan and report actions without copying files")
	flag.StringVar(&hfToken, "huggingface-token", "", "Optional HuggingFace API token (falls back to HF_API_TOKEN)")
	flag.StringVar(&civitaiToken, "civitai-token", "", "Optional Civitai API token (falls back to CIVITAI_API_TOKEN)")
	flag.Parse()

	var meta metadataIndex
	var err error
	if metadataPath != "" {
		meta, err = loadMetadataFile(metadataPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to load metadata: %v\n", err)
			os.Exit(1)
		}
	} else {
		meta = metadataIndex{}
	}

	organizer := newOrganizer(sourceDir, destinationDir, dryRun, meta)
	organizer.HuggingFaceToken = hfToken
	organizer.CivitAIToken = civitaiToken

	manifest, err := organizer.run()
	if err != nil {
		fmt.Fprintf(os.Stderr, "%v\n", err)
		os.Exit(1)
	}

	if dryRun {
		payload, _ := json.MarshalIndent(manifestToMap(manifest), "", "  ")
		fmt.Println(string(payload))
		return
	}
}
