# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


from pathlib import Path
import sys
import scrapy

downloads = Path.home() / 'Downloads'

url = 'https://www.nba.com/stats/draft/history/?Season='
frames = pd.read_html(url)

process = CrawlerProcess(settings={
    "FEEDS": {
        "items.json": {"format": "json"},
    },
})


process.crawl(MySpider)
process.start()