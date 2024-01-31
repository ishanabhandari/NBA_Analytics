import os
import time
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError

SEASONS = list(range(2022, 2023))
NBA_DATA = 'nba_data'
STANDINGS = os.path.join(NBA_DATA, 'standings')
SCORES = os.path.join(NBA_DATA, 'scores')


async def get_nba_data_html(url, selector, sleep=8, retries=3):
    html = None
    for i in range(1, retries + 1):
        time.sleep(sleep * (i + 1))
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                await page.goto(url)
                print(await page.title())
                html = await page.inner_html(selector=selector)
                await browser.close()
        except PlaywrightTimeoutError:
            print(f'Timeout error on {url}')
            continue
        else:
            break
    return html


async def scrape_season(season):
    url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games.html"
    html = await get_nba_data_html(url, "#content .filter")

    soup = BeautifulSoup(html, features="html.parser")
    links = soup.find_all("a")
    standings_pages = [f"https://www.basketball-reference.com{l['href']}" for l in links]

    for url in standings_pages:
        save_path = os.path.join(STANDINGS, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_nba_data_html(url, "#all_schedule")
        with open(save_path, "w+", encoding="utf-8") as f:
            f.write(html)


async def scrape_game(standings_file):
    with open(standings_file, 'r') as f:
        html = f.read()

    soup = BeautifulSoup(html, features="html.parser")
    links = soup.find_all("a")
    hrefs = [l.get('href') for l in links]
    box_scores = [f"https://www.basketball-reference.com{l}" for l in hrefs if l and "boxscore" in l and '.html' in l]

    for url in box_scores:
        save_path = os.path.join(SCORES, url.split("/")[-1])
        if os.path.exists(save_path):
            continue

        html = await get_nba_data_html(url, "#content")
        if not html:
            continue
        with open(save_path, "w+", encoding="utf-8") as f:
            f.write(html)


async def main():
    standings_files = []
    for season in SEASONS:
        await scrape_season(season)
        standings_files = os.listdir(STANDINGS)
        for standings_file in standings_files:
            await scrape_game(os.path.join(STANDINGS, standings_file))

    for season in SEASONS:
        files = [f for f in standings_files if str(season) in f]
        for f in files:
            filepath = os.path.join(STANDINGS, f)
            await scrape_game(filepath)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
