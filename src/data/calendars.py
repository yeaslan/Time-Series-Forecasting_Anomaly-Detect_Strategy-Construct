from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd


@dataclass
class TradingCalendar:
    name: str = "XNYS"
    timezone: str = "America/New_York"
    open_time: str = "09:30"
    close_time: str = "16:00"

    def trading_days(self, start: str | datetime, end: str | datetime) -> pd.DatetimeIndex:
        start = pd.Timestamp(start, tz=self.timezone).normalize()
        end = pd.Timestamp(end, tz=self.timezone).normalize()
        all_days = pd.date_range(start=start, end=end, freq="B")
        return all_days.tz_convert("UTC")

    def trading_minutes(self, start: str | datetime, end: str | datetime, freq: str = "1T") -> pd.DatetimeIndex:
        calendar_days = self.trading_days(start, end)
        minutes = []
        for day in calendar_days:
            session_open = day.tz_convert(self.timezone).replace(
                hour=int(self.open_time.split(":")[0]), minute=int(self.open_time.split(":")[1])
            )
            session_close = day.tz_convert(self.timezone).replace(
                hour=int(self.close_time.split(":")[0]), minute=int(self.close_time.split(":")[1])
            )
            session_range = pd.date_range(
                start=session_open, end=session_close, freq=freq, tz=self.timezone, inclusive="left"
            )
            minutes.append(session_range.tz_convert("UTC"))
        if minutes:
            return minutes[0].append(minutes[1:])  # type: ignore[arg-type]
        return pd.DatetimeIndex([], tz="UTC")


def align_to_calendar(
    data: pd.DataFrame,
    calendar: TradingCalendar,
    freq: str = "B",
    on: Optional[str] = None,
    groupby: Optional[str] = None,
) -> pd.DataFrame:
    ts_col = on or "timestamp"
    df = data.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

    if groupby:
        aligned_frames: list[pd.DataFrame] = []
        for key, subdf in df.groupby(groupby):
            full_index = calendar.trading_days(subdf[ts_col].min(), subdf[ts_col].max()) if freq == "B" else calendar.trading_minutes(subdf[ts_col].min(), subdf[ts_col].max(), freq=freq)
            subdf = subdf.set_index(ts_col).reindex(full_index)
            subdf[groupby] = key
            aligned_frames.append(subdf.reset_index().rename(columns={"index": ts_col}))
        return pd.concat(aligned_frames, ignore_index=True)

    full_index = calendar.trading_days(df[ts_col].min(), df[ts_col].max()) if freq == "B" else calendar.trading_minutes(df[ts_col].min(), df[ts_col].max(), freq=freq)
    return df.set_index(ts_col).reindex(full_index).reset_index().rename(columns={"index": ts_col})
