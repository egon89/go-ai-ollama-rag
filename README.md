# go-ai-ollama-rag

## Resources
- [Bill's Guide to A.I - YouTube](https://www.youtube.com/playlist?list=PLADD_vxzPcZDzTmmub99S0Ne58ApvJZjJ)
- [Ardan Labs - Github](https://github.com/ardanlabs/ai-training/tree/main)

## Ollama - docker
[Docker configuration](https://github.com/egon89/docker-mono-repo/tree/main/ollama)

- llama3
- nomic-embed-text

## Run

Start the Ollama server according [Ollama - docker](https://github.com/egon89/docker-mono-repo/tree/main/ollama).


In the root directory of the project, run:
```bash
docker compose up -d
```

Put a file in the `documents` folder, then run:
```bash
go run cmd/context/main.go
```
