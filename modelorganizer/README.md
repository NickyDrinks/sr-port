# Model Organizer or Ultra File Juggler 2004 Pro (Midi Edition)

This directory contains the standalone Go-based model organization utility. It is structured so it can be copied into its own Git repository without pulling in the rest of `sr-port`.

## Usage as a separate repository

1. Copy the contents of this folder into a new repository (for example `git init` inside the copied folder).
2. Adjust the module name in `go.mod` if you plan to publish it under a different import path.
3. Build or test the utility as usual:

   ```bash
   go build
   go test ./...
   ```

The CLI itself is implemented in `main.go`, with accompanying tests in `main_test.go`.

## Running the organizer

See `--help` for command-line flags:

```bash
go run . --help
```

The tool expects a source directory of model files (safetensors, gguf, etc.) and produces a structured layout with hashed metadata sidecars and optional HuggingFace/CivitAI enrichment.
