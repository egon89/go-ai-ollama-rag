package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/egon89/go-ai-ollama-rag/mongodb"
	"github.com/tmc/langchaingo/llms/ollama"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

var embeddingPath = "./embeddings"
var printChunks = false

type document struct {
	ID        string    `json:"id"`
	FileName  string    `json:"fileName"`
	Text      string    `json:"text"`
	Embedding []float32 `json:"embedding"`
}

func main() {
	file := "./documents/Saga-Knight-Level-8-Ao-80-Tibia-Life.pdf"
	fileName := filepath.Base(file)

	rawText, err := extractRawText(file)
	if err != nil {
		log.Fatalf("Error extracting raw text: %v", err)
	}

	chunks := ChunkText(rawText, 1000)
	log.Printf("Total chunks: %d\n", len(chunks))
	if printChunks {
		fmt.Println("--------------------------------------------------")
		fmt.Println("Text Chunks:")
		for i, chunk := range chunks {
			fmt.Println("--------------------------------------------------")
			fmt.Printf("Chunk %d: %s\n", i+1, chunk)
		}
	}

	err = createEmbeddings(fileName, chunks)
	if err != nil {
		log.Fatalf("Error creating embeddings: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	collection, err := setupDatabase(ctx)
	if err != nil {
		log.Fatalf("Error setup database: %v", err)
	}

	if err := insertEmbeddings(ctx, collection); err != nil {
		log.Fatalf("insert embedding error: %v", err)
	}
}

func extractRawText(filePath string) (string, error) {
	file, err := os.ReadFile(filePath)
	if err != nil {
		return "", err
	}

	req, err := http.NewRequest("PUT", "http://localhost:9998/tika", bytes.NewReader(file))
	if err != nil {
		return "", err
	}

	req.Header.Set("Accept", "text/plain") // <- ensures plain text response
	req.Header.Set("Content-Type", "application/octet-stream")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected response code: %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	return RemoveInvalidUTF8(RemovePUA(string(body))), nil
}

func RemoveInvalidUTF8(s string) string {
	valid := make([]rune, 0, len(s))
	for len(s) > 0 {
		r, size := utf8.DecodeRuneInString(s)
		if r == utf8.RuneError && size == 1 {
			// Skip invalid rune
			s = s[size:]
			continue
		}
		valid = append(valid, r)
		s = s[size:]
	}
	return string(valid)
}

func RemovePUA(s string) string {
	var b strings.Builder
	for _, r := range s {
		if isPUA(r) {
			continue
		}
		b.WriteRune(r)
	}
	return b.String()
}

func isPUA(r rune) bool {
	return (r >= 0xE000 && r <= 0xF8FF) ||
		(r >= 0xF0000 && r <= 0xFFFFD) ||
		(r >= 0x100000 && r <= 0x10FFFD)
}

func ChunkText(text string, chunkSize int) []string {
	var chunks []string
	for i := 0; i < len(text); i += chunkSize {
		end := i + chunkSize
		if end > len(text) {
			end = len(text)
		}
		chunks = append(chunks, text[i:end])
	}
	return chunks
}

func createEmbeddings(fileName string, texts []string) error {
	embeddingFile := fmt.Sprintf("%s/%s.embeddings", embeddingPath, fileName)

	if _, err := os.Stat(embeddingFile); err == nil {
		fmt.Printf("Embedding path %s already exists, skipping creation.\n", embeddingFile)
		return nil
	}

	llm, err := ollama.New(
		ollama.WithModel("nomic-embed-text"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		return fmt.Errorf("failed to create LLM client: %w", err)
	}

	file, err := os.Create(embeddingFile)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()

	for i, text := range texts {
		embedding, err := llm.CreateEmbedding(context.Background(), []string{text})
		if err != nil {
			return fmt.Errorf("failed to create embedding for text chunk %d: %w", i, err)
		}

		doc := document{
			ID:        fmt.Sprintf("doc-%d", i),
			FileName:  fileName,
			Text:      text,
			Embedding: embedding[0],
		}

		data, err := json.Marshal(doc)
		if err != nil {
			return fmt.Errorf("failed to marshal document: %w", err)
		}

		if _, err := file.Write(data); err != nil {
			return fmt.Errorf("failed to write document to file: %w", err)
		}

		if _, err := file.WriteString("\n"); err != nil {
			return fmt.Errorf("failed to write newline to file: %w", err)
		}

		fmt.Print("\033[u\033[K")
		fmt.Printf("Successfully processed chunk %d", i)

	}

	fmt.Println()
	log.Printf("Embeddings saved to %s", embeddingFile)

	return nil
}

func setupDatabase(ctx context.Context) (*mongo.Collection, error) {
	client, err := mongodb.Connect(ctx, "localhost:27017", "root", "root")
	if err != nil {
		return nil, fmt.Errorf("connectToMongo: %w", err)
	}

	const dbName = "go-ai-rag"
	const collectionName = "context"

	db := client.Database(dbName)

	// Create database and collection.
	collection, err := mongodb.CreateCollection(ctx, db, collectionName)
	if err != nil {
		return nil, fmt.Errorf("createCollection: %w", err)
	}

	log.Println("Created Collection")

	const indexName = "vector_index"
	settings := mongodb.VectorIndexSettings{
		NumDimensions: 1024,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	// Create vector index.
	if err := mongodb.CreateVectorIndex(ctx, collection, indexName, settings); err != nil {
		return nil, fmt.Errorf("createVectorIndex: %w", err)
	}

	log.Println("Created Vector Index")

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "id", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}

	// Create a unique index for the document.
	collection.Indexes().CreateOne(ctx, indexModel)

	log.Println("Created Unique Index")

	return collection, nil
}

func insertEmbeddings(ctx context.Context, collection *mongo.Collection) error {
	input, err := os.Open("./embeddings/Saga-Knight-Level-8-Ao-80-Tibia-Life.pdf.embeddings")
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer input.Close()

	var counter int

	fmt.Print("\n")
	fmt.Print("\033[s")

	scanner := bufio.NewScanner(input)
	for scanner.Scan() {
		counter++

		// get document from the file
		doc := scanner.Text()

		fmt.Print("\033[u\033[K")
		fmt.Printf("Insering Data: %d", counter)

		var d document
		if err := json.Unmarshal([]byte(doc), &d); err != nil {
			return fmt.Errorf("unmarshal: %w", err)
		}

		// check if the document is already in the database
		res := collection.FindOne(ctx, bson.D{{Key: "id", Value: d.ID}})
		if res.Err() == nil {
			continue
		}

		if !errors.Is(res.Err(), mongo.ErrNoDocuments) {
			return fmt.Errorf("another error: %w", err)
		}

		if _, err := collection.InsertOne(ctx, d); err != nil {
			return fmt.Errorf("insert: %w", err)
		}
	}

	return nil
}
