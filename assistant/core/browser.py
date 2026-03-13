

"""Browser management using Playwright."""
import base64
import asyncio
from playwright.async_api import async_playwright, Page, Playwright, Browser, BrowserContext

class BrowserManager:
    def __init__(self):
        self._playwright: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def get_page(self) -> Page:
        if not self._playwright:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(headless=True)
            self._context = await self._browser.new_context(
                viewport={"width": 1280, "height": 800}
            )
            self._page = await self._context.new_page()
            # Default timeout for element waiting
            self._page.set_default_timeout(10000)
        return self._page

    async def close(self):
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def get_screenshot_base64(self) -> str:
        page = await self.get_page()
        # Small delay to allow renders
        await asyncio.sleep(0.5)
        screenshot_bytes = await page.screenshot(type="jpeg", quality=60)
        base64_image = base64.b64encode(screenshot_bytes).decode("utf-8")
        print(f"[DEBUG] Screenshot captured, base64 length: {len(base64_image)}")
        return base64_image

browser_manager = BrowserManager()

async def save_screenshot_to_file(self, filename: str = None) -> str:
    """Save screenshot to a file in the screenshots directory."""
    import os
    page = await self.get_page()
    await asyncio.sleep(0.5)
    screenshot_bytes = await page.screenshot(type="jpeg", quality=60)
    if filename is None:
        from datetime import datetime
        filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
    screenshots_dir = os.path.join(os.path.dirname(__file__), "../screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)
    file_path = os.path.join(screenshots_dir, filename)
    with open(file_path, "wb") as f:
        f.write(screenshot_bytes)
    print(f"[DEBUG] Screenshot saved to: {file_path}")
    return file_path