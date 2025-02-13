# Darwin Experimentation Architecture

This repository contains the experimental architecture for the Darwin project, exploring intelligent systems using multiple LLM providers and architectural patterns.

## 🚀 Features

- Integration with multiple LLM providers:
  - OpenAI GPT models
  - Anthropic Claude
  - Amazon Bedrock
  - Ollama (local models)
- Flexible architecture for experimentation
- Comprehensive testing suite
- CI/CD pipeline with automated testing
- Discord integration for development notifications

## 📋 Prerequisites

- Python 3.10 or higher
- API keys for:
  - OpenAI
  - Anthropic
  - AWS (for Bedrock)
- Local Ollama installation (optional)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd Architecture
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys and configuration

## 🧪 Running Tests

```bash
pytest
```

For test coverage:
```bash
pytest --cov=src
```

## 🏗️ Project Structure

```
Architecture/
├── docs/              # Documentation
├── src/               # Source code
│   └── calculator/    # Example module
├── tests/             # Test files
├── scripts/           # Utility scripts
└── .github/           # GitHub Actions workflows
```

## 🔄 CI/CD Pipeline

The project includes a GitHub Actions pipeline that:
- Runs automated tests
- Generates test coverage reports
- Sends notifications to Discord about build status

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests
4. Submit a pull request

## 🔒 Security

- Never commit API keys or sensitive data
- Use environment variables for sensitive information
- Store production secrets in GitHub Secrets

## 📝 License

[Your License Here]

## 🔗 Related Projects

- [Main Darwin Repository]
- [Other Related Projects]

## 📞 Contact

[Your Contact Information]

---
*Note: This is an experimental architecture for the Darwin project. Features and structure may change as the project evolves.*