from logger import logger

def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Разбивает текст на чанки длиной примерно chunk_size символов
    с перекрытием overlap символов.
    """
    logger.info(f"Старт чанкования: текст длиной {len(text)} символов, "
                f"чанк_size={chunk_size}, overlap={overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    if not text:
        logger.warning("Передан пустой текст для чанкования")
        return []

    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        logger.debug(f"Чанк {len(chunks)}: позиции {start}-{end}, длина {len(chunk)}")

        if end == text_length:
            break

        # следующий старт с учётом перекрытия
        start = end - overlap
    logger.info(f"Чанкование завершено, получено {len(chunks)} частей")
    return chunks