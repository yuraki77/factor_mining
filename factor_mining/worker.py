from __future__ import annotations

from apscheduler.schedulers.blocking import BlockingScheduler

from factor_mining.config import load_settings
from factor_mining.data.binance import BinanceArchiveClient
from factor_mining.storage import MetadataStore


def run_worker() -> None:
    settings = load_settings()
    store = MetadataStore(settings.data.sqlite_path)
    client = BinanceArchiveClient(settings, store)
    scheduler = BlockingScheduler(timezone="UTC")

    @scheduler.scheduled_job("cron", hour=3, minute=15)
    def daily_data_sync() -> None:
        client.sync()

    scheduler.start()

