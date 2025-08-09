# My Life Chat App

A personal AI chat companion that remembers and helps you find information about your life using LangGraph, PostgreSQL with vector embeddings, and Ollama.

## 🎯 What is this?

This is a hobby project that creates your own personal AI assistant - think of it as a digital memory bank for your life. Ask it about anything from your past experiences, preferences, or memories, and it'll search through your stored knowledge to give you relevant answers.

## ✨ Key Features

- **🤖 AI Chat Interface**: Natural language conversations about your personal life
- **🧠 Knowledge Base**: Store and retrieve personal information using vector embeddings
- **🔍 Semantic Search**: Find relevant memories even with fuzzy queries
- **🌐 Web Interface**: Clean, modern UI for chatting
- **📊 Knowledge Management**: Add and organize your personal data
- **🚀 Real-time**: WebSocket support for instant responses

## 🏗️ Architecture

Built with a modern AI stack:
- **LangGraph** for intelligent workflow orchestration
- **PostgreSQL + pgvector** for semantic vector storage
- **Ollama** for local LLM inference
- **FastAPI** for the web API
- **Docker** for easy deployment

## 🚀 Quick Start

1. **Clone and start**:
   ```bash
   git clone <your-repo>
   cd my-life-chat-app
   make init
   ```

2. **Pull the AI models**:
   ```bash
   make pull-models
   ```

3. **Add some knowledge** (optional):
   ```bash
   make seed
   ```

4. **Start chatting**:
   - Open http://localhost:8000
   - Ask about your life!

## 💡 Usage Examples

- "What restaurants do I like in San Francisco?"
- "When did I last go hiking?"
- "What are my favorite books?"
- "Tell me about my trip to Japan"

## 🛠️ Development

### Prerequisites
- Docker and Docker Compose
- Python 3.10+ (for local development)

### Local Development
```bash
# Start services
make init

# View logs
make logs

# Restart everything
make restart

# Stop services
make stop
```

### Project Structure
```
my-life-chat-app/
├── main.py              # Core LangGraph application
├── seed.py              # Sample data loader
├── docker-compose.yml   # Service orchestration
├── Dockerfile          # App container
├── Makefile            # Development commands
└── requirements.txt    # Python dependencies
```

## 🔧 Configuration

### Environment Variables
- `DATABASE_URL`: PostgreSQL connection (default: postgresql://user:password@localhost:5432/my_personal_data)
- `OLLAMA_URL`: Ollama service URL (default: http://ollama:11434)

### LLM Models
The app uses:
- `llama3` for chat responses
- `mxbai-embed-large` for embeddings

## 🆘 Troubleshooting

**Database issues?**
```bash
make restart
```

**Models not found?**
```bash
make pull-models
```

**View logs:**
```bash
make logs
```

---

*Built with ❤️ as a hobby project to explore AI, vector databases, and personal knowledge management.*
