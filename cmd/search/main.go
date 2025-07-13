package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/egon89/go-ai-ollama-rag/mongodb"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
)

type searchResult struct {
	// ID        int       `bson:"id"`
	ID        string    `json:"id"`
	Text      string    `bson:"text"`
	Embedding []float64 `bson:"embedding"`
	Score     float64   `bson:"score"`
}

var printResult = false

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Ask a question: ")

	question, _ := reader.ReadString('\n')
	if question == "" {
		return nil
	}

	fmt.Print("THIS MAY TAKE A MINUTE OR MORE, BE PATIENT\n\n")

	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()

	results, err := vectorSearch(ctx, question)
	if err != nil {
		return fmt.Errorf("vectorSearch: %w", err)
	}

	if err := questionResponse(ctx, question, results); err != nil {
		return fmt.Errorf("questionResponse: %w", err)
	}

	return nil
}

func vectorSearch(ctx context.Context, question string) ([]searchResult, error) {

	// -------------------------------------------------------------------------
	// Use ollama to generate a vector embedding for the question.

	// Open a connection with ollama to access the model.
	llm, err := ollama.New(ollama.WithModel("nomic-embed-text"))
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}

	// Get the vector embedding for the question.
	embedding, err := llm.CreateEmbedding(context.Background(), []string{question})
	if err != nil {
		return nil, fmt.Errorf("create embedding: %w", err)
	}

	// -------------------------------------------------------------------------
	// Establish a connection with mongo and access the collection.

	// Connect to mongodb.
	client, err := mongodb.Connect(ctx, "localhost:27017", "root", "root")
	if err != nil {
		return nil, fmt.Errorf("connectToMongo: %w", err)
	}

	const dbName = "go-ai-rag"
	const collectionName = "context"

	// Capture a connection to the collection. We assume this exists with
	// data already.
	col := client.Database(dbName).Collection(collectionName)

	// -------------------------------------------------------------------------
	// Perform the vector search.

	// We want to find the nearest neighbors from the question vector embedding.
	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":         "vector_index",
				"exact":         false,
				"path":          "embedding",
				"queryVector":   embedding[0],
				"numCandidates": 5,
				"limit":         5,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"id":        1,
				"text":      1,
				"embedding": 1,
				"score": bson.M{
					"$meta": "vectorSearchScore",
				},
			}},
		},
	}

	cur, err := col.Aggregate(ctx, pipeline)
	if err != nil {
		return nil, fmt.Errorf("aggregate: %w", err)
	}
	defer cur.Close(ctx)

	var results []searchResult
	if err := cur.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("all: %w", err)
	}

	return results, nil
}

func questionResponse(ctx context.Context, question string, results []searchResult) error {

	// Open a connection with ollama to access the model.
	llm, err := ollama.New(ollama.WithModel("llama3.2"))
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	// Format a prompt to direct the model what to do with the content and
	// the question.
	prompt := `Use the following pieces of information to answer the user's question.
	
If you don't know the answer, say that you don't know.	

Answer the question and provide additional helpful information.

Responses should be properly formatted to be easily read.
	
Context: %s
	
Question: %s
`

	var wordCount int
	var chunks strings.Builder
	var validResultCounter int

	for _, res := range results {
		if res.Score >= .70 {
			validResultCounter++

			chunks.WriteString(res.Text)
			chunks.WriteString(".\n")

			// We don't need to provide more than 1000 words.
			words := strings.Fields(res.Text)
			wordCount += len(words)

			if printResult {
				fmt.Println("---")
				fmt.Printf("score: %v\n", res.Score)
				fmt.Println(res.Text)
				fmt.Println("---")
			}

			if wordCount >= 1000 {
				break
			}
		}
	}

	fmt.Printf("Results found: %d\n", validResultCounter)

	content := chunks.String()
	if content == "" {
		fmt.Println("Don't have enough information to provide an answer")
		return nil
	}

	finalPrompt := fmt.Sprintf(prompt, content, question)

	// Setup a wait group to wait for the entire response.
	var wg sync.WaitGroup
	wg.Add(1)

	// This function will display the response as it comes from the server.
	f := func(ctx context.Context, chunk []byte) error {
		if ctx.Err() != nil || len(chunk) == 0 {
			wg.Done()
			return nil
		}

		fmt.Printf("%s", chunk)
		return nil
	}

	// Send the prompt to the model server.
	if _, err := llm.Call(ctx, finalPrompt, llms.WithStreamingFunc(f)); err != nil {
		return fmt.Errorf("call: %w", err)
	}

	// Wait until we receive the entire response.
	wg.Wait()

	return nil
}
