# URL to Markdown with Crawl4AI and Render

Convert any webpage to clean, LLM-friendly markdown. Powered by [Crawl4AI](https://github.com/unclecode/crawl4ai).

## Overview

LLMs work best with clean text, but the web is full of messy HTML, navigation menus, ads, and JavaScript-rendered content. This template gives you a simple API that crawls any URL—including pages that require JavaScript to load—and converts the content into clean markdown ready to feed into GPT-4, Claude, or any other model.

It includes a minimal web UI so you can try it out immediately—just paste a URL and see the markdown output. For production use, call the REST API from your code to integrate web content into your LLM pipelines.

## Endpoints

| Endpoint       | Method | Description                                |
| -------------- | ------ | ------------------------------------------ |
| `/`            | GET    | Web UI for converting URLs to markdown     |
| `/health`      | GET    | Health check                               |
| `/docs`        | GET    | Interactive API documentation (Swagger UI) |
| `/crawl`       | POST   | Crawl a single URL and return markdown     |
| `/crawl/batch` | POST   | Crawl multiple URLs concurrently (max 10)  |

## Usage

### Crawl a single URL

```bash
curl -X POST https://your-service.onrender.com/crawl \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

**Response:**

```json
{
  "url": "https://example.com",
  "title": "Example Domain",
  "markdown": "# Example Domain\n\nThis domain is for use in illustrative examples...",
  "word_count": 42,
  "success": true
}
```

### Crawl with options

```bash
curl -X POST https://your-service.onrender.com/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://news.ycombinator.com",
    "include_raw": true,
    "filter_threshold": 0.5
  }'
```

**Request options:**

| Field               | Type   | Default  | Description                                |
| ------------------- | ------ | -------- | ------------------------------------------ |
| `url`               | string | required | The URL to crawl                           |
| `include_raw`       | bool   | `false`  | Include unfiltered raw markdown            |
| `filter_threshold`  | float  | `0.4`    | Content filter aggressiveness (0-1)        |
| `wait_for_selector` | string | `null`   | CSS selector to wait for before extraction |
| `js_code`           | string | `null`   | JavaScript to execute before extraction    |

### Crawl multiple URLs

```bash
curl -X POST https://your-service.onrender.com/crawl/batch \
  -H "Content-Type: application/json" \
  -d '[
    "https://example.com",
    "https://httpbin.org/html"
  ]'
```

### Handle dynamic content

For pages that load content via JavaScript:

```bash
curl -X POST https://your-service.onrender.com/crawl \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/dynamic-page",
    "wait_for_selector": ".content-loaded",
    "js_code": "document.querySelector(\"button.load-more\").click()"
  }'
```

## Deployment

1. Click **Use this template** to create your own copy of this repository
1. In your [Render Dashboard](https://dashboard.render.com/), create a new **Blueprint**
1. Connect your new repository
1. Render automatically detects and deploys from `render.yaml`

## Environment variables

| Variable | Description             | Default |
| -------- | ----------------------- | ------- |
| `PORT`   | Port for the web server | `8000`  |

## Local development

```bash
# Clone the repository
git clone https://github.com/render-examples/crawl4ai.git
cd crawl4ai

# Build and run with Docker
docker build -t url-to-markdown .
docker run -p 8000:8000 url-to-markdown

# Visit http://localhost:8000 for the web UI
```

## Next steps

Once you have this running, here are some ideas for what you could build:

- **Add an LLM layer** - Pipe the markdown into GPT-4 or Claude to summarize, extract entities, or answer questions about the content
- **Build a RAG pipeline** - Store crawled content in a vector database and build a chatbot that answers questions from your sources
- **Create a monitoring service** - Schedule regular crawls to track changes on competitor sites, documentation, or news sources
- **Power an AI agent** - Let your agent browse the web by calling this API to gather information for complex tasks
- **Generate datasets** - Collect domain-specific content for fine-tuning models or building evaluation sets

Crawl4AI also supports [LLM-based extraction](https://docs.crawl4ai.com/core/extraction-strategies/) if you want structured data instead of markdown.

## Resources

- [Crawl4AI documentation](https://docs.crawl4ai.com/)
- [Crawl4AI GitHub](https://github.com/unclecode/crawl4ai)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [Render Docker documentation](https://render.com/docs/docker)
