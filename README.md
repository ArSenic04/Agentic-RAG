# AI Research Agent System

## Overview
This is an advanced AI-powered research agent that leverages multiple AI technologies to conduct comprehensive research and generate detailed answers to user queries.

## Features
- Web search using Tavily Search API
- Web content retrieval and processing
- Advanced text embedding and vector search
- Multi-agent workflow with research and drafting stages
- Fallback mechanism for error handling

## Prerequisites
- Python 3.8+
- API Keys:
  - Tavily API Key
  - Groq API Key

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-research-agent.git
cd ai-research-agent
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project root:
```
TAVILY_API_KEY=your_tavily_api_key
GROQ_API_KEY=your_groq_api_key
```

## Usage
```python
from research_agent import run_research_system

query = "What is quantum computing?"
answer = run_research_system(query)
print(answer)
```

## Components
- Research Agent: Searches web, retrieves content
- Answer Drafting Agent: Synthesizes research into a comprehensive answer
- Fallback Agent: Provides response if primary agents fail

## License
[Specify your license, e.g., MIT]

## Contributing
Contributions are welcome! Please read the contributing guidelines.
