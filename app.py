"""
URL to Markdown API - Powered by Crawl4AI
A simple web service that crawls URLs and returns clean, LLM-friendly markdown.
"""

import os
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, HttpUrl, Field

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy


# Global crawler instance
crawler: Optional[AsyncWebCrawler] = None

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage crawler lifecycle."""
    global crawler
    browser_config = BrowserConfig(
        headless=True,
        java_script_enabled=True,
    )
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.__aenter__()
    yield
    await crawler.__aexit__(None, None, None)


app = FastAPI(
    title="URL to Markdown API",
    description="Convert any webpage to clean, LLM-friendly markdown using Crawl4AI",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CrawlRequest(BaseModel):
    """Request body for crawling a URL."""
    url: HttpUrl = Field(..., description="The URL to crawl")
    include_raw: bool = Field(
        default=False,
        description="Include raw markdown in addition to filtered content"
    )
    filter_threshold: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Content filter threshold (0-1). Higher = more aggressive filtering"
    )
    wait_for_selector: Optional[str] = Field(
        default=None,
        description="CSS selector to wait for before extracting content"
    )
    js_code: Optional[str] = Field(
        default=None,
        description="JavaScript code to execute before extraction"
    )


class DeepCrawlRequest(BaseModel):
    """Request body for deep crawling."""
    url: str = Field(..., description="The starting URL to crawl")
    max_pages: int = Field(default=10, ge=1, le=100, description="Maximum pages to crawl")
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum depth to follow links")


class CrawlResponse(BaseModel):
    """Response from crawling a URL."""
    url: str
    title: Optional[str] = None
    markdown: str
    raw_markdown: Optional[str] = None
    word_count: int
    success: bool
    error: Optional[str] = None


class DeepCrawlResponse(BaseModel):
    """Response from deep crawling."""
    success: bool
    starting_url: str
    max_pages: int
    max_depth: int
    pages_crawled: int
    pages: list
    combined_markdown: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str


# Create static directory if it doesn't exist
STATIC_DIR.mkdir(exist_ok=True)


# Create index.html for the web UI
INDEX_HTML = '''<!DOCTYPE html>
<html>
<head>
    <title>Crawl4AI - Web Crawler</title>
    <style>
        body{font-family:Arial;max-width:900px;margin:50px auto;padding:20px;background:#1a1a2e;color:white;}
        input,textarea,button{padding:12px;margin:10px 0;border-radius:8px;border:none;font-size:16px;}
        input,textarea{width:100%;background:#0f3460;color:white;}
        button{background:#e94560;color:white;cursor:pointer;}
        pre{background:#0f3460;padding:15px;border-radius:8px;overflow-x:auto;}
    </style>
</head>
<body>
    <h1>🕷️ Crawl4AI Web Crawler</h1>
    <p>Convert any webpage to clean markdown, ready for LLMs.</p>
    
    <h3>Single URL Scrape</h3>
    <input type="text" id="singeUrl" placeholder="Enter URL">
    <button onclick="singleCrawl()">Scrape</button>
    
    <h3>Deep Crawl (follows links)</h3>
    <input type="text" id="deepUrl" placeholder="Starting URL">
    <input type="number" id="maxPages" placeholder="Max pages" value="10">
    <input type="number" id="maxDepth" placeholder="Max depth" value="2">
    <button onclick="deepCrawl()">Deep Crawl</button>
    
    <pre id="result">Waiting...</pre>

    <script>
        async function singleCrawl() {
            const url = document.getElementById('singeUrl').value;
            if (!url) { alert('Enter URL'); return; }
            document.getElementById('result').innerText = 'Scraping...';
            try {
                const response = await fetch('/crawl', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url: url})
                });
                const data = await response.json();
                document.getElementById('result').innerText = JSON.stringify(data, null, 2);
            } catch(e) { document.getElementById('result').innerText = 'Error: ' + e.message; }
        }
        
        async function deepCrawl() {
            const url = document.getElementById('deepUrl').value;
            const maxPages = document.getElementById('maxPages').value;
            const maxDepth = document.getElementById('maxDepth').value;
            if (!url) { alert('Enter URL'); return; }
            document.getElementById('result').innerText = 'Deep crawling... This may take a minute';
            try {
                const response = await fetch('/crawl/deep', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({url: url, max_pages: parseInt(maxPages), max_depth: parseInt(maxDepth)})
                });
                const data = await response.json();
                document.getElementById('result').innerText = JSON.stringify(data, null, 2);
            } catch(e) { document.getElementById('result').innerText = 'Error: ' + e.message; }
        }
    </script>
</body>
</html>'''

# Create the index.html file
with open(STATIC_DIR / "index.html", "w") as f:
    f.write(INDEX_HTML)


@app.get("/", include_in_schema=False)
async def root():
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="url-to-markdown",
        version="1.0.0"
    )


@app.post("/crawl", response_model=CrawlResponse)
async def crawl_url(request: CrawlRequest):
    """Crawl a single URL and return its content as markdown."""
    if crawler is None:
        raise HTTPException(status_code=503, detail="Crawler not initialized")
    
    try:
        md_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=request.filter_threshold,
                threshold_type="fixed"
            )
        )
        
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=md_generator,
            wait_for=request.wait_for_selector,
            js_code=[request.js_code] if request.js_code else None,
        )
        
        result = await crawler.arun(url=str(request.url), config=run_config)
        
        if not result.success:
            return CrawlResponse(
                url=str(request.url),
                markdown="",
                word_count=0,
                success=False,
                error=result.error_message or "Crawl failed"
            )
        
        fit_markdown = result.markdown.fit_markdown if hasattr(result.markdown, 'fit_markdown') else result.markdown
        raw_markdown = result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else str(result.markdown)
        
        if isinstance(fit_markdown, str):
            content = fit_markdown
        else:
            content = str(fit_markdown) if fit_markdown else raw_markdown
        
        word_count = len(content.split()) if content else 0
        
        return CrawlResponse(
            url=str(request.url),
            title=result.metadata.get("title") if result.metadata else None,
            markdown=content,
            raw_markdown=raw_markdown if request.include_raw else None,
            word_count=word_count,
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/crawl/deep", response_model=DeepCrawlResponse)
async def crawl_deep(request: DeepCrawlRequest):
    """
    Deep crawl a website starting from a URL.
    Automatically follows links to discover and crawl multiple pages.
    """
    if crawler is None:
        raise HTTPException(status_code=503, detail="Crawler not initialized")
    
    try:
        config = CrawlerRunConfig(
            deep_crawl_strategy=BFSDeepCrawlStrategy(
                max_depth=request.max_depth,
                max_pages=request.max_pages,
                include_external=False,
            ),
            cache_mode=CacheMode.BYPASS,
        )
        
        result = await crawler.arun(url=request.url, config=config)
        
        pages = []
        if hasattr(result, 'crawled_pages'):
            for page in result.crawled_pages:
                pages.append({
                    "url": page.url,
                    "depth": getattr(page, 'depth', 0),
                    "title": page.metadata.get("title", "") if hasattr(page, 'metadata') and page.metadata else "",
                    "markdown_length": len(page.markdown) if hasattr(page, 'markdown') else 0
                })
        
        combined_markdown = ""
        if hasattr(result, 'markdown'):
            combined_markdown = result.markdown if isinstance(result.markdown, str) else str(result.markdown)
        
        return DeepCrawlResponse(
            success=True,
            starting_url=request.url,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            pages_crawled=len(pages),
            pages=pages,
            combined_markdown=combined_markdown,
            error=None
        )
        
    except Exception as e:
        return DeepCrawlResponse(
            success=False,
            starting_url=request.url,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            pages_crawled=0,
            pages=[],
            combined_markdown="",
            error=str(e)
        )


@app.post("/crawl/batch", response_model=list[CrawlResponse])
async def crawl_batch(urls: list[HttpUrl]):
    """
    Crawl multiple URLs concurrently and return markdown for each.
    Limited to 10 URLs per request to prevent abuse.
    """
    if crawler is None:
        raise HTTPException(status_code=503, detail="Crawler not initialized")
    
    if len(urls) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 URLs per batch request")
    
    results = []
    
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.4, threshold_type="fixed")
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
    )
    
    try:
        crawl_results = await crawler.arun_many(
            urls=[str(u) for u in urls],
            config=run_config
        )
        
        for result in crawl_results:
            if result.success:
                fit_markdown = result.markdown.fit_markdown if hasattr(result.markdown, 'fit_markdown') else result.markdown
                raw_markdown = result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else str(result.markdown)
                content = fit_markdown if isinstance(fit_markdown, str) else str(fit_markdown) if fit_markdown else raw_markdown
                
                results.append(CrawlResponse(
                    url=result.url,
                    title=result.metadata.get("title") if result.metadata else None,
                    markdown=content,
                    word_count=len(content.split()) if content else 0,
                    success=True
                ))
            else:
                results.append(CrawlResponse(
                    url=result.url,
                    markdown="",
                    word_count=0,
                    success=False,
                    error=result.error_message or "Crawl failed"
                ))
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
