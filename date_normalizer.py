"""
Алгоритмическая нормализация относительных дат.
Это критично для стабильности: LLM каждый раз может по-разному
интерпретировать "в следующую среду", а Python — всегда одинаково.
"""
import re
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

WEEKDAYS_RU = {
    "понедельник": 0, "пн": 0,
    "вторник": 1, "вт": 1,
    "среда": 2, "ср": 2, "среду": 2,
    "четверг": 3, "чт": 3,
    "пятница": 4, "пт": 4, "пятницу": 4,
    "суббота": 5, "сб": 5, "субботу": 5,
    "воскресенье": 6, "вс": 6,
}

MONTHS_RU = {
    "январ": 1, "феврал": 2, "март": 3, "марта": 3,
    "апрел": 4, "ма": 5, "июн": 6,
    "июл": 7, "август": 8, "авг": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}


def normalize_date(text: str, meeting_date: str) -> str | None:
    """
    Переводит относительное выражение в YYYY-MM-DD.
    Возвращает None, если не удалось.
    meeting_date — в формате YYYY-MM-DD.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip().lower()
    md = datetime.strptime(meeting_date, "%Y-%m-%d")

    # Уже в нужном формате
    if re.match(r"^\d{4}-\d{2}-\d{2}$", text):
        return text

    # "завтра"
    if text == "завтра":
        return (md + timedelta(days=1)).strftime("%Y-%m-%d")

    # "послезавтра"
    if text == "послезавтра":
        return (md + timedelta(days=2)).strftime("%Y-%m-%d")

    # "сегодня"
    if text == "сегодня":
        return md.strftime("%Y-%m-%d")

    # "в <день недели>"
    m = re.match(r"^в\s+(понедельник|вторник|среду?|четверг|пятницу?|субботу?|воскресенье|пн|вт|ср|чт|пт|сб|вс)$", text)
    if m:
        target_wd = WEEKDAYS_RU[m.group(1)]
        days_ahead = target_wd - md.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (md + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "на следующей неделе" — берём понедельник следующей недели
    if "следующ" in text and "недел" in text:
        days_ahead = 7 - md.weekday()
        return (md + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "в конце недели" — воскресенье текущей недели
    if "конц" in text and "недел" in text:
        days_ahead = 6 - md.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return (md + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # "через N день/дня/дней/неделю/недели/недель/месяц/месяца"
    m = re.match(r"^через\s+(\d+|-?\w+)\s+(день|дня|дней|неделю|недели|недель|месяц|месяца|мес)$", text)
    if m:
        num_str, unit = m.group(1), m.group(2)
        try:
            num = int(num_str)
        except ValueError:
            num_map = {"одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5}
            num = num_map.get(num_str)
        if num is None:
            return None
        if "дн" in unit:
            return (md + timedelta(days=num)).strftime("%Y-%m-%d")
        if "недел" in unit:
            return (md + timedelta(weeks=num)).strftime("%Y-%m-%d")
        if "месяц" in unit or "мес" == unit:
            return (md + relativedelta(months=num)).strftime("%Y-%m-%d")

    # "через неделю/месяц" (без числа)
    if re.match(r"^через\s+неделю$", text):
        return (md + timedelta(weeks=1)).strftime("%Y-%m-%d")
    if re.match(r"^через\s+месяц$", text):
        return (md + relativedelta(months=1)).strftime("%Y-%m-%d")

    # "<число> <месяца>" — например "1 июля", "15 апреля"
    m = re.match(r"^(\d{1,2})\s+(январ|феврал|март|марта|апрел|ма[яй]?|июн|июл|август|авг|сентябр|октябр|ноябр|декабр)", text)
    if m:
        day = int(m.group(1))
        month_prefix = m.group(2)
        month = MONTHS_RU.get(month_prefix)
        if month:
            year = md.year
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                return None

    # "в начале/середине/конце <месяца>"
    m = re.match(r"^в\s+(начале|середине|конце)\s+(январ|феврал|март|марта|апрел|ма[яй]?|июн|июл|август|авг|сентябр|октябр|ноябр|декабр)", text)
    if m:
        pos, month_prefix = m.group(1), m.group(2)
        month = MONTHS_RU.get(month_prefix)
        if month:
            if "начал" in pos:
                day = 3
            elif "сер" in pos:
                day = 15
            else:
                # конец месяца — последний день
                if month == 12:
                    next_m = datetime(md.year + 1, 1, 1)
                else:
                    next_m = datetime(md.year, month + 1, 1)
                day = (next_m - timedelta(days=1)).day
            try:
                return datetime(md.year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                return None

    return None


def normalize_tasks_dates(tasks: list[dict], meeting_date: str) -> list[dict]:
    """
    Применяет нормализацию дат ко всем задачам.
    Если срок уже в YYYY-MM-DD — оставляет как есть.
    Если срок текстовый — пытается нормализовать.
    Если не получилось — ставит null.
    """
    result = []
    for task in tasks:
        task = dict(task)  # копия
        raw_deadline = task.get("срок")
        if raw_deadline is None or raw_deadline == "" or raw_deadline == "null":
            task["срок"] = None
        else:
            normalized = normalize_date(str(raw_deadline), meeting_date)
            task["срок"] = normalized
        result.append(task)
    return result