# define the model here and the loader and the search script both pick it up.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# must match the vector(N) setting in db/migrations/001_create_tables.sql
EXPECTED_DIMENSION = 384