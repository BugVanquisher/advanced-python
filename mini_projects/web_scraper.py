"""
Async Web Scraper Mini Project
==============================
A simple async web scraper that demonstrates concurrent HTTP requests,
rate limiting, and structured data extraction.
"""

import asyncio
import aiohttp
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import json


@dataclass
class ScrapedPage:
    url: str
    title: Optional[str]
    status_code: int
    content_length: int
    response_time: float
    error: Optional[str] = None


class AsyncWebScraper:
    """
    Async web scraper with rate limiting and concurrent requests.
    """
    
    def __init__(self, max_concurrent: int = 10, delay: float = 0.1):
        self.max_concurrent = max_concurrent
        self.delay = delay
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'AsyncScraper/1.0'}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_page(self, url: str) -> ScrapedPage:
        """Scrape a single page with rate limiting."""
        async with self.semaphore:
            start_time = time.time()
            
            try:
                async with self.session.get(url) as response:
                    content = await response.text()
                    response_time = time.time() - start_time
                    
                    # Extract title (simple regex approach)
                    title = None
                    title_start = content.find('<title>')
                    if title_start != -1:
                        title_end = content.find('</title>', title_start)
                        if title_end != -1:
                            title = content[title_start + 7:title_end].strip()
                    
                    result = ScrapedPage(
                        url=url,
                        title=title,
                        status_code=response.status,
                        content_length=len(content),
                        response_time=response_time
                    )
                    
            except Exception as e:
                response_time = time.time() - start_time
                result = ScrapedPage(
                    url=url,
                    title=None,
                    status_code=0,
                    content_length=0,
                    response_time=response_time,
                    error=str(e)
                )
            
            # Rate limiting
            await asyncio.sleep(self.delay)
            return result
    
    async def scrape_urls(self, urls: List[str]) -> List[ScrapedPage]:
        """Scrape multiple URLs concurrently."""
        tasks = [self.scrape_page(url) for url in urls]
        return await asyncio.gather(*tasks)
    
    async def scrape_sitemap(self, base_url: str, max_pages: int = 50) -> List[ScrapedPage]:
        """
        Simple sitemap discovery and scraping.
        This is a basic implementation - in practice you'd want proper sitemap parsing.
        """
        # Try common sitemap locations
        sitemap_urls = [
            urljoin(base_url, '/sitemap.xml'),
            urljoin(base_url, '/sitemap_index.xml'),
            urljoin(base_url, '/robots.txt')
        ]
        
        # For demo purposes, we'll just scrape some common paths
        parsed_url = urlparse(base_url)
        common_paths = [
            '/', '/about', '/contact', '/blog', '/products', 
            '/services', '/team', '/news', '/help', '/faq'
        ]
        
        urls_to_scrape = [
            f"{parsed_url.scheme}://{parsed_url.netloc}{path}" 
            for path in common_paths[:max_pages]
        ]
        
        return await self.scrape_urls(urls_to_scrape)


class ScrapingReporter:
    """Generate reports from scraping results."""
    
    @staticmethod
    def generate_summary(results: List[ScrapedPage]) -> Dict:
        """Generate summary statistics."""
        total_pages = len(results)
        successful = len([r for r in results if r.error is None])
        failed = total_pages - successful
        
        successful_results = [r for r in results if r.error is None]
        avg_response_time = (
            sum(r.response_time for r in successful_results) / len(successful_results)
            if successful_results else 0
        )
        
        return {
            'total_pages': total_pages,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_pages if total_pages > 0 else 0,
            'avg_response_time': avg_response_time,
            'total_content_length': sum(r.content_length for r in successful_results)
        }
    
    @staticmethod
    def save_results(results: List[ScrapedPage], filename: str):
        """Save results to JSON file."""
        data = []
        for result in results:
            data.append({
                'url': result.url,
                'title': result.title,
                'status_code': result.status_code,
                'content_length': result.content_length,
                'response_time': result.response_time,
                'error': result.error
            })
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)


async def demo_scraper():
    """Demo function showing scraper usage."""
    
    # Example URLs to scrape (using httpbin for testing)
    test_urls = [
        'https://httpbin.org/delay/1',
        'https://httpbin.org/status/200',
        'https://httpbin.org/status/404',
        'https://httpbin.org/html',
        'https://httpbin.org/json'
    ]
    
    print("🕷️  Starting async web scraper demo...")
    
    async with AsyncWebScraper(max_concurrent=3, delay=0.5) as scraper:
        start_time = time.time()
        
        # Scrape individual URLs
        results = await scraper.scrape_urls(test_urls)
        
        total_time = time.time() - start_time
        
        # Generate and display summary
        summary = ScrapingReporter.generate_summary(results)
        
        print(f"\n📊 Scraping Summary:")
        print(f"   Total pages: {summary['total_pages']}")
        print(f"   Successful: {summary['successful']}")
        print(f"   Failed: {summary['failed']}")
        print(f"   Success rate: {summary['success_rate']:.2%}")
        print(f"   Avg response time: {summary['avg_response_time']:.2f}s")
        print(f"   Total time: {total_time:.2f}s")
        
        print(f"\n📄 Individual Results:")
        for result in results:
            status = "✅" if result.error is None else "❌"
            print(f"   {status} {result.url}")
            if result.title:
                print(f"      Title: {result.title}")
            if result.error:
                print(f"      Error: {result.error}")
            print(f"      Status: {result.status_code}, Time: {result.response_time:.2f}s")
        
        # Save results
        ScrapingReporter.save_results(results, 'scraping_results.json')
        print(f"\n💾 Results saved to scraping_results.json")


if __name__ == "__main__":
    # Run the demo
    asyncio.run(demo_scraper())