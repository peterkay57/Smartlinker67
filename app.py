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


class CrawlResponse(BaseModel):
    """Response from crawling a URL."""
    url: str
    title: Optional[str] = None
    markdown: str
    raw_markdown: Optional[str] = None
    word_count: int
    success: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str


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
    """
    Crawl a URL and return its content as markdown.
    
    The markdown is optimized for LLM consumption with optional content filtering
    to remove boilerplate, navigation, and other non-essential content.
    """
    if crawler is None:
        raise HTTPException(status_code=503, detail="Crawler not initialized")
    
    try:
        # Configure markdown generator with content filter
        md_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(
                threshold=request.filter_threshold,
                threshold_type="fixed"
            )
        )
        
        # Build crawler config
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            markdown_generator=md_generator,
            wait_for=request.wait_for_selector,
            js_code=[request.js_code] if request.js_code else None,
        )
        
        # Execute crawl
        result = await crawler.arun(url=str(request.url), config=run_config)
        
        if not result.success:
            return CrawlResponse(
                url=str(request.url),
                markdown="",
                word_count=0,
                success=False,
                error=result.error_message or "Crawl failed"
            )
        
        # Extract markdown content
        fit_markdown = result.markdown.fit_markdown if hasattr(result.markdown, 'fit_markdown') else result.markdown
        raw_markdown = result.markdown.raw_markdown if hasattr(result.markdown, 'raw_markdown') else str(result.markdown)
        
        # Handle case where fit_markdown might be the same object
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
    
    # Use default config for batch
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
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
