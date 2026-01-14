# Birdwatch

A personal Twitter/X archive explorer with semantic search, clustering, and AI-powered Q&A.

## Features

- **Import Twitter Archives**: Upload your Twitter data export and browse your tweet history
- **Full-text Search**: Search tweets using PostgreSQL full-text search
- **Semantic Search**: Find related tweets using vector embeddings (OpenAI text-embedding-3-small)
- **Smart Clustering**: Automatically group tweets by topic using UMAP + HDBSCAN
- **AI Q&A**: Ask natural language questions about your tweets using Claude

## Prerequisites

- [Pixi](https://pixi.sh/) - Package manager (handles Python, PostgreSQL, etc.)
- OpenAI API key (for embeddings)
- Anthropic API key (for AI Q&A features)

## Quick Start

### 1. Clone and Setup

```bash
git clone <your-repo-url>
cd birdwatch

# Install dependencies with pixi
pixi install
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env and add your API keys:
# - OPENAI_API_KEY=sk-...
# - ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Initialize Database

```bash
# Initialize PostgreSQL data directory and create databases
pixi run db-init

# Apply schema
pixi run db-schema
```

### 4. Start the Application

```bash
# Start the database (if not running)
pixi run db-start

# Start the web server
pixi run dev
```

Open http://localhost:8001 in your browser.

## Usage

### Importing Your Twitter Archive

1. Request your data from Twitter/X (Settings > Your Account > Download an archive)
2. Once received, go to the Import page in Birdwatch
3. Upload the ZIP file
4. The importer will parse tweets and store them in the database

### Generating Embeddings

After importing tweets, generate embeddings for semantic search:

```bash
# This runs automatically when you access the clustering features
# Or trigger manually through the web UI
```

### Running Clustering

Clustering groups your tweets by topic automatically:

```bash
# Access via the Explore page in the web UI
```

## Development

### Project Structure

```
birdwatch/
├── app/
│   ├── db/           # Database queries and connection pool
│   ├── routes/       # FastAPI route handlers
│   ├── services/     # Business logic (embeddings, clustering, etc.)
│   └── templates/    # Jinja2 HTML templates
├── scripts/          # Utility scripts
├── tests/            # Test files
├── .env.example      # Environment template
├── pixi.toml         # Pixi dependencies and tasks
└── pyproject.toml    # Python package config
```

### Available Commands

```bash
pixi run dev          # Start dev server with hot reload
pixi run test         # Run tests
pixi run db-start     # Start PostgreSQL
pixi run db-stop      # Stop PostgreSQL
pixi run db-init      # Initialize database (first time only)
pixi run db-schema    # Apply database schema
```

### Database

Birdwatch uses PostgreSQL with the following extensions:
- `pgvector` - Vector similarity search for embeddings
- `pg_trgm` - Trigram matching for fuzzy text search

The database runs on port **5434** (to avoid conflicts with system PostgreSQL).

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `OPENAI_API_KEY` | OpenAI API key for embeddings | Yes |
| `ANTHROPIC_API_KEY` | Anthropic API key for AI Q&A | Yes |
| `MEDIA_DIR` | Directory for media files | No (default: `media`) |
| `DEBUG` | Enable debug mode | No (default: `false`) |

## Tech Stack

- **Backend**: FastAPI, Python 3.11
- **Database**: PostgreSQL 16 with pgvector
- **ML/AI**: OpenAI embeddings, Anthropic Claude, UMAP, HDBSCAN
- **Frontend**: Jinja2 templates, HTMX, Tailwind CSS
