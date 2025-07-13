package mongodb

type Index struct {
	Id   string `bson:"id"`
	Type string `bson:"type"`
}

type VectorIndexSettings struct {
	NumDimensions int
	Path          string
	Similarity    string
}
