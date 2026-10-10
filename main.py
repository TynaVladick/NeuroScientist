"""
RAG-система: Ollama (генерация + embeddings) + FAISS + LangChain.

Зависимости:
    pip install langchain-core langchain-community langchain-ollama \
        langchain-text-splitters faiss-cpu numpy chardet requests \
        pypdf python-docx beautifulsoup4

Запуск:
    python rag_system.py                    # документы и индекс: <корень проекта>/documents
    python rag_system.py --rebuild          # принудительно пересоздать индекс

Команды в диалоге:
    /files              — список проиндексированных файлов
    /file <имя>         — «закрепить» файл: на все следующие вопросы модель читает его
                          (самые подходящие к вопросу разделы + краткая сводка статьи)
    /unfile             — снять закрепление
    /summarize          — подготовить краткие сводки ВСЕХ статей (один раз; кэшируется, можно прервать)
    /summarize <имя>    — сводка одной статьи
    /review [тема]      — зонтичный обзор: сводки статей по теме → единый структурированный обзор
                          (/review без темы или /review all — по всем статьям)
    /clear              — сбросить историю диалога
    exit                — выход

Как это работает:
  * Сводки. Статья целиком в окно не влезает, поэтому из неё берутся ключевые разделы
    (аннотация, методы, результаты, обсуждение, выводы) и локальная модель делает из них
    компактную карточку (~170 слов). Карточки хранятся в documents/summaries.json и
    пересоздаются только для изменившихся файлов.
  * Зонтичный обзор. Выбираются статьи, ближайшие к теме → берутся их карточки →
    если они не помещаются в окно, сжимаются группами (map-reduce) → итоговый обзор.
    Вопросы вроде «сделай обзор по всем статьям…» распознаются автоматически.
  * Вопрос по конкретной статье. Файл определяется по имени в вопросе / закреплению /
    лучшему совпадению; из него берутся самые подходящие фрагменты + карточка статьи.
    Если и этого мало, модель может ответить [[ЧИТАТЬ: файл | что искать]].

Индекс пересоздаётся автоматически, если файлы в папке документов изменились
или поменялись параметры индексации.
"""
from __future__ import annotations

import os
os.environ["NO_PROXY"] = "127.0.0.1,localhost"
os.environ["no_proxy"] = "127.0.0.1,localhost"

import argparse
import json
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple
from urllib.parse import unquote

import chardet
import numpy as np
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

DEFAULT_HOST = "http://127.0.0.1:11434"
# Корень проекта = папка, где лежит этот скрипт (не зависит от того, откуда запущен Python)
PROJECT_ROOT = Path(__file__).resolve().parent
DOCS_DIR = PROJECT_ROOT / "documents"  # здесь лежат и документы, и индекс FAISS
REVIEWS_DIR = PROJECT_ROOT / "reviews"  # сюда сохраняются обзоры (вне папки документов!)
INDEX_NAME = "vector_store"  # файлы индекса: vector_store.faiss и vector_store.pkl
MANIFEST_NAME = f"{INDEX_NAME}.manifest.json"  # «отпечаток» документов, по которым собран индекс
SUMMARIES_NAME = "summaries.json"  # кэш кратких сводок статей
SUMMARY_VERSION = 1  # увеличить при смене формата/промпта сводки
INDEX_FORMAT_VERSION = 3  # увеличить, если меняется логика нарезки/эмбеддингов
SUPPORTED_FORMATS = (".txt", ".md", ".markdown", ".pdf", ".docx", ".html", ".htm", ".xml")
MARKDOWN_FORMATS = (".md", ".markdown")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("rag")


# ──────────────────────────────── Конфигурация ────────────────────────────────
@dataclass
class RAGConfig:
    model_name: str = "gemma4"
    embedding_model: str = "nomic-embed-text"
    ollama_host: str = DEFAULT_HOST

    # --- Генерация ---
    # num_ctx — общее окно контекста (промпт + ответ). Если промпт в него не влезает,
    # Ollama МОЛЧА отбрасывает начало промпта, поэтому контекст обрезается под бюджет.
    num_ctx: int = 8192
    num_predict: int = -1  # максимум токенов ответа; -1 = без ограничения
    answer_reserve_tokens: int = 2048  # сколько токенов окна оставить под ответ, если num_predict = -1
    chars_per_token: float = 2.5  # консервативная оценка для русского текста
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    keep_alive: str = "30m"  # держать модель в памяти между вопросами
    think: bool = False  # режим «размышлений» (для сводок и обзоров только тратит время)

    # --- Индексация и поиск ---
    chunk_size: int = 1500
    chunk_overlap: int = 200
    min_chunk_chars: int = 40  # чанки короче (например, одни заголовки) не индексируются
    top_k: int = 6  # сколько фрагментов отдаём модели
    fetch_k: int = 24  # из скольких кандидатов MMR выбирает разнообразные
    use_mmr: bool = True
    embed_batch_size: int = 32
    follow_links: bool = True  # подтягивать файлы, на которые ссылаются найденные фрагменты
    max_linked_files: int = 3

    # --- Фокус на файле ---
    focus_chunks: int = 10  # сколько самых подходящих фрагментов брать из фокусного файла
    focus_share: float = 0.65  # какую долю бюджета контекста отдать фокусным файлам
    max_focus_files: int = 2
    allow_file_requests: bool = True  # модель может сама запросить файл через [[ЧИТАТЬ: ...]]
    max_listed_files: int = 40  # сколько имён файлов показывать модели в списке

    # --- Сводки и обзоры ---
    summary_input_chars: int = 9000  # сколько текста статьи (ключевые разделы) отдаём на сводку
    summary_max_tokens: int = 450  # длина сводки
    review_max_articles: int = 25  # сколько статей брать в обзор по теме
    review_answer_tokens: int = 2500  # резерв окна под итоговый обзор
    review_map_tokens: int = 600  # длина промежуточных заметок при сжатии групп

    # --- Диалог ---
    max_history_turns: int = 3  # сколько последних пар «вопрос-ответ» помнить
    history_answer_chars: int = 600  # длинные ответы в истории урезаются


# ──────────────────────────────── Клиент Ollama ───────────────────────────────
class OllamaClient:
    """
    Тонкий клиент /api/generate на requests. Заменяет langchain OllamaLLM: у него в связке
    с новыми pydantic падает «'ModelPrivateAttr' object has no attribute 'generate'»,
    а своя реализация короче, быстрее и не зависит от версий.
    """

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self.url = f"{cfg.ollama_host.rstrip('/')}/api/generate"
        self.session = requests.Session()

    def stream(
        self,
        prompt: str,
        num_predict: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Iterator[str]:
        c = self.cfg
        payload = {
            "model": c.model_name,
            "prompt": prompt,
            "stream": True,
            "think": c.think,
            "keep_alive": c.keep_alive,
            "options": {
                "num_ctx": c.num_ctx,
                "num_predict": c.num_predict if num_predict is None else num_predict,
                "temperature": c.temperature if temperature is None else temperature,
                "top_p": c.top_p,
                "repeat_penalty": c.repeat_penalty,
            },
        }
        with self.session.post(self.url, json=payload, stream=True, timeout=(10, 900)) as resp:
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:300]}")
            for line in resp.iter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                if data.get("response"):
                    yield data["response"]
                if data.get("done"):
                    break

    def generate(
        self, prompt: str, num_predict: Optional[int] = None, temperature: Optional[float] = None
    ) -> str:
        return "".join(self.stream(prompt, num_predict, temperature)).strip()


# ──────────────────────────────── Загрузчики ──────────────────────────────────
def normalize_text(text: str) -> str:
    """Единые переводы строк (Windows \\r\\n ломали разбиение по абзацам) и без нулевых байтов."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "").replace("\ufeff", "")


def read_text_file(path: Path) -> str:
    """Читает текст, быстро определяя кодировку (chardet смотрит только начало файла)."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")  # покрывает и обычный UTF-8
    except UnicodeDecodeError:
        encoding = chardet.detect(raw[:100_000])["encoding"] or "cp1251"
        try:
            text = raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            text = raw.decode("cp1251", errors="replace")
    return normalize_text(text)


def load_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader  # type: ignore
    reader = PdfReader(str(path))
    return normalize_text("\n".join((page.extract_text() or "") for page in reader.pages))


def load_docx(path: Path) -> str:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return normalize_text("\n".join(parts))


def load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(read_text_file(path), "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return normalize_text(soup.get_text(separator="\n", strip=True))


def load_xml(path: Path) -> str:
    """Парсит XML встроенным ElementTree (кодировку берёт из объявления в файле)."""
    root = ET.parse(str(path)).getroot()
    lines: List[str] = []

    def walk(el: ET.Element, depth: int) -> None:
        tag = el.tag.split("}")[-1] if isinstance(el.tag, str) else ""
        text = (el.text or "").strip()
        if tag and text:
            lines.append(f"{'  ' * depth}{tag}: {text}")
        for child in el:
            walk(child, depth + 1)
            tail = (child.tail or "").strip()
            if tail:
                lines.append(f"{'  ' * (depth + 1)}{tail}")

    walk(root, 0)
    return normalize_text("\n".join(lines))


LOADERS: Dict[str, Callable[[Path], str]] = {
    ".txt": read_text_file,
    ".md": read_text_file,
    ".markdown": read_text_file,
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".html": load_html,
    ".htm": load_html,
    ".xml": load_xml,
}


# ──────────────────────────────── Промпты ─────────────────────────────────────
PROMPT = PromptTemplate.from_template(
    """Ты — внимательный исследователь-аналитик. Отвечай на вопрос, опираясь на контекст из документов.

Правила:
- Отвечай подробно и развёрнуто: раскрывай тему полностью, приводи детали, цифры, определения, примеры и пояснения из контекста.
- Длинный ответ структурируй: краткое вступление, основные пункты, вывод.
- Не выдумывай. Если нужной информации в контексте нет, прямо скажи об этом и ответь только на ту часть, которая подтверждается документами.
- Отвечай на языке вопроса.
{read_rule}
История диалога:
{history}

{files}Контекст:
{context}

Вопрос: {question}

Подробный ответ:"""
)

# Правило «запроси файл»: модель отвечает одной строкой, программа подгружает файл и спрашивает заново
READ_RULE = (
    "- Если в контексте нет нужных данных (например, видны только метаданные статьи: название, "
    "авторы, журнал), но они могут быть в полном тексте одного из файлов из списка «Доступные файлы», "
    "ответь ТОЛЬКО одной строкой, без пояснений и без другого текста: "
    "[[ЧИТАТЬ: имя_файла | что именно искать]]\n"
)
_READ_RE = re.compile(
    r"^\s*\[\[\s*(?:ЧИТАТЬ|READ)\s*:\s*([^|\]]+?)\s*(?:\|\s*([^\]]*?))?\s*\]\]",
    re.IGNORECASE | re.DOTALL,
)

SUMMARY_PROMPT = """Ты помогаешь готовить обзор литературы. Составь краткую структурную карточку научной статьи по её ключевым фрагментам.

Формат (строго, на русском, всего не более 170 слов, без вступлений и пояснений):
Тип и дизайн: ...
Выборка / материал: ...
Метод / вмешательство: ...
Главные результаты: (с ключевыми цифрами, если есть)
Выводы авторов: ...
Ограничения: ...

Не выдумывай: если пункта нет в тексте — пиши «не указано».

{text}

Карточка:"""

MAP_PROMPT = """Ниже — карточки научных статей (или промежуточные заметки), пронумерованные [N]. Сожми их в заметки для будущего обзора по теме «{topic}».
Сохрани номера [N] у каждого факта, дизайн исследований, ключевые цифры и расхождения между работами. Сгруппируй по смыслу. Не выдумывай. Не более 300 слов.

{notes}

Заметки:"""

REVIEW_PROMPT = """Ты — исследователь, готовящий зонтичный обзор (umbrella review): обобщение нескольких научных работ по общей теме. Ниже — карточки статей, пронумерованные [N].

Тема / запрос: {topic}

Правила:
- Опирайся только на карточки; не выдумывай. Если данных нет — так и скажи.
- Каждое утверждение сопровождай ссылками на статьи вида [3] или [2, 5].
- Не смешивай уровни доказательности: указывай тип работы (систематический обзор, РКИ, когортное, пилотное, модельное и т. п.).
- Если статьи слабо связаны между собой или с темой — скажи об этом прямо и сгруппируй их по тематикам, а не придумывай общий вывод.
- Отвечай на русском, структурированно.

Структура ответа:
1. Объём и состав: сколько работ, какие дизайны, популяции, области.
2. Основные выводы по темам или исходам (группируй работы).
3. Согласованность и противоречия между работами.
4. Качество и ограничения доказательств (только по тому, что видно в карточках; формальную оценку риска систематической ошибки не выдавай).
5. Пробелы в знаниях и направления дальнейших исследований.
6. Общий вывод (3–5 предложений).

Карточки статей:
{notes}

Зонтичный обзор:"""

# Ссылки на другие markdown-файлы: (file.md), [[file]], «file.md»
_MD_LINK_RE = re.compile(r"[^\s()\[\]<>\"'|]+?\.(?:md|markdown)\b", re.IGNORECASE)
_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+)")

# Вопросы-«обзоры» распознаются автоматически
_OVERVIEW_RE = re.compile(
    r"зонтичн|umbrella|обзор\w*\s+(?:всех|по\s+всем|статей|литератур|исследован)"
    r"|(?:по|во|в)\s+всем\s+(?:статьям|документам|файлам)|сравни\w*\s+(?:все|статьи|исследования)",
    re.IGNORECASE,
)


def is_overview_question(question: str) -> bool:
    return bool(_OVERVIEW_RE.search(question))


# ──────────────────────────────── Сводки статей ───────────────────────────────
_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.*)$", re.MULTILINE)
_SECTION_KEYS = {
    "abstract": ("abstract", "summary", "аннотац", "резюме"),
    "methods": ("method", "material", "design", "метод", "материал"),
    "results": ("result", "finding", "результат"),
    "discussion": ("discussion", "обсужден"),
    "conclusion": ("conclusion", "заключ", "вывод"),
    "limitations": ("limitation", "ограничен"),
}
# Доля бюджета текста, отдаваемая каждому разделу (в порядке подачи модели)
_SECTION_SHARE = {
    "abstract": 0.30, "methods": 0.15, "results": 0.25,
    "discussion": 0.10, "conclusion": 0.12, "limitations": 0.08,
}


def extract_title(text: str, filename: str = "?") -> str:
    """Название: title из YAML front matter, иначе первый заголовок #, иначе имя файла."""
    head = text[:3000]
    m = re.match(r"\A---\n(.*?)\n---\n", head, re.DOTALL)
    if m:
        t = re.search(r"^title:\s*(.+)$", m.group(1), re.MULTILINE | re.IGNORECASE)
        if t:
            return t.group(1).strip().strip("\"'")
    m = re.search(r"^#\s+(.+)$", head, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return Path(filename).stem


def split_sections(text: str) -> List[Tuple[str, str]]:
    """
    [(класс, текст)] по markdown-заголовкам. Первый элемент — «шапка» до первого заголовка
    (класс 'front'). Подзаголовки наследуют класс родителя: «Background/Methods» внутри
    Abstract остаются аннотацией.
    """
    matches = list(_HEADING_RE.finditer(text))
    out: List[Tuple[str, str]] = [("front", text[: matches[0].start()] if matches else text)]
    cls, lvl = "", 0
    for i, m in enumerate(matches):
        level, title = len(m.group(1)), m.group(2).lower()
        found = next((k for k, keys in _SECTION_KEYS.items() if any(s in title for s in keys)), "")
        inherited = False
        if cls == "abstract" and level > lvl:
            found, inherited = "abstract", True
        elif not found and cls and level > lvl:
            found, inherited = cls, True
        if not inherited:
            cls, lvl = found, level
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out.append((found, text[m.start():end].strip()))
    return out


def build_summary_input(text: str, filename: str, limit: int) -> str:
    """Выжимка статьи для сводки: название + ключевые разделы в пределах limit символов."""
    title = f"Название: {extract_title(text, filename)}"
    sections = split_sections(text)
    buckets: Dict[str, List[str]] = {}
    for cls, body in sections[1:]:
        if cls:
            buckets.setdefault(cls, []).append(body)

    if not buckets:  # нет узнаваемых разделов: начало + конец
        if len(text) <= limit:
            return f"{title}\n\n{text}"
        half = limit // 2
        return f"{title}\n\n{text[:half]}\n[…]\n{text[-half:]}"

    front = sections[0][1].strip()[:600]
    rest = max(limit - len(front) - len(title), 1000)
    parts = [title, front]
    for cls, share in _SECTION_SHARE.items():
        if cls in buckets:
            parts.append("\n".join(buckets[cls])[: int(rest * share)])
    return "\n\n".join(p for p in parts if p)


class PrefixedOllamaEmbeddings(OllamaEmbeddings):
    """
    nomic-embed-text обучен с префиксами задач: без них качество поиска заметно хуже.
    Для остальных моделей префиксы пустые — поведение не меняется.
    """

    doc_prefix: str = ""
    query_prefix: str = ""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return super().embed_documents([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> List[float]:
        return super().embed_query(self.query_prefix + text)


# ──────────────────────────────── RAG-система ─────────────────────────────────
class RAGSystem:
    def __init__(self, config: Optional[RAGConfig] = None, store_dir: str | Path = DOCS_DIR):
        self.cfg = config or RAGConfig()
        self.store_dir = Path(store_dir)
        os.environ["OLLAMA_HOST"] = self.cfg.ollama_host

        self._check_ollama_connection()

        self.llm = OllamaClient(self.cfg)
        is_nomic = "nomic" in self.cfg.embedding_model.lower()
        self.embeddings = PrefixedOllamaEmbeddings(
            model=self.cfg.embedding_model,
            base_url=self.cfg.ollama_host,
            doc_prefix="search_document: " if is_nomic else "",
            query_prefix="search_query: " if is_nomic else "",
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # Markdown режем по заголовкам; сами заголовки остаются в тексте раздела
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
            strip_headers=False,
        )
        self.vector_store: Optional[FAISS] = None
        # Матрица нормированных векторов чанков: по ней фокус на файле и ранжирование статей
        # считаются на numpy без повторных запросов к Ollama
        self.mat: Optional[np.ndarray] = None
        self.row_docs: List[Document] = []
        self.file_rows: Dict[str, np.ndarray] = {}
        self.file_paths: Dict[str, Path] = {}
        self.known_files: Dict[str, str] = {}  # имя файла в нижнем регистре -> настоящее имя
        self.files_line: str = ""  # список файлов для промпта
        self.pinned_files: List[str] = []  # файлы, закреплённые командой /file
        self.summaries: Dict[str, dict] = {}  # имя файла -> {"fp": [size, mtime], "text": ...}
        self.history: List[tuple[str, str]] = []

    # ---------- Подключение ----------
    def _check_ollama_connection(self, retries: int = 3, delay: float = 2.0) -> None:
        host = self.cfg.ollama_host
        log.info(f"Проверка подключения к Ollama на {host}...")
        last_error: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(f"{host}/api/tags", timeout=5)
                resp.raise_for_status()
                names = {m["name"] for m in resp.json().get("models", [])}
                log.info(f"✓ Ollama доступен. Моделей: {len(names)}")
                self._warn_if_missing(names)
                return
            except (requests.RequestException, ValueError) as e:
                last_error = e
                if attempt < retries:
                    log.info(f"✗ Нет ответа, повтор через {delay:.0f}с ({attempt}/{retries})...")
                    time.sleep(delay)

        raise ConnectionError(
            f"Ollama недоступен на {host}: {last_error}\n"
            "Запустите приложение Ollama или выполните: ollama serve\n"
            f"Загрузка моделей: ollama pull {self.cfg.model_name} && "
            f"ollama pull {self.cfg.embedding_model}"
        )

    def _warn_if_missing(self, installed: set[str]) -> None:
        bases = {n.split(":")[0] for n in installed}
        for model in (self.cfg.model_name, self.cfg.embedding_model):
            if model not in installed and model not in bases:
                log.warning(f"⚠ Модель '{model}' не найдена. Выполните: ollama pull {model}")

    # ---------- Загрузка документов ----------
    @staticmethod
    def _is_indexable(path: Path, supported_formats: tuple[str, ...]) -> bool:
        return (
            path.is_file()
            and path.suffix.lower() in supported_formats
            and not path.name.startswith(("~$", "."))  # временные/скрытые файлы
        )

    def load_document(self, file_path: str | Path) -> str:
        path = Path(file_path)
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            log.info(f"Формат {path.suffix} не поддерживается, читаю как TXT...")
            loader = read_text_file
        return loader(path)

    def load_documents_from_directory(
        self, directory_path: str, supported_formats: tuple[str, ...] = SUPPORTED_FORMATS
    ) -> List[Document]:
        directory = Path(directory_path)
        if not directory.exists():
            log.error(f"Директория {directory_path} не существует!")
            return []

        documents: List[Document] = []
        for path in sorted(directory.rglob("*")):
            if not self._is_indexable(path, supported_formats):
                continue
            try:
                content = self.load_document(path)
            except ImportError as e:
                log.error(f"✗ {path.name}: не хватает библиотеки ({e})")
                continue
            except Exception as e:
                log.error(f"✗ Ошибка при загрузке {path.name}: {e}")
                continue

            if not content.strip():
                log.warning(f"⚠ {path.name}: пустой документ, пропускаю")
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "file_type": path.suffix.lower(),
                    },
                )
            )
            log.info(f"✓ Загружен: {path.name} ({len(content):,} символов)")
        return documents

    # ---------- Нарезка на чанки ----------
    def _split_markdown(self, doc: Document) -> List[Document]:
        """
        Режет markdown по разделам. Каждый чанк подписан названием документа и путём
        заголовков — так фрагмент «Результаты» из статьи сам знает, из какой он статьи.
        """
        title = extract_title(doc.page_content, doc.metadata.get("filename", "?"))
        chunks: List[Document] = []
        for section in self.md_splitter.split_text(doc.page_content):
            heading_path = " > ".join(
                section.metadata[k] for k in ("h1", "h2", "h3", "h4") if k in section.metadata
            )
            for piece in self.text_splitter.split_text(section.page_content):
                body = "\n".join(
                    line for line in piece.splitlines() if not line.lstrip().startswith("#")
                ).strip()
                if len(body) < self.cfg.min_chunk_chars:
                    continue  # заголовок без содержимого — только засоряет поиск
                header = f"[Документ: {title}]"
                if heading_path:
                    header += f"\n[Раздел: {heading_path}]"
                chunks.append(Document(page_content=f"{header}\n{piece}", metadata=dict(doc.metadata)))
        return chunks

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            file_chunks: List[Document] = []
            if doc.metadata.get("file_type") in MARKDOWN_FORMATS:
                file_chunks = self._split_markdown(doc)
            if not file_chunks:  # не markdown или очень короткий файл — режем обычным способом
                file_chunks = self.text_splitter.split_documents([doc])
            chunks.extend(file_chunks)
        return chunks

    # ---------- Векторное хранилище ----------
    def create_vector_store(self, documents: List[Document]) -> None:
        if not documents:
            log.error("Нет документов для обработки!")
            return

        chunks = self.split_documents(documents)
        if not chunks:
            log.error("После нарезки не осталось ни одного фрагмента!")
            return
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i  # сквозной номер: внутри файла возрастает по ходу текста

        per_file = Counter(c.metadata["filename"] for c in chunks)
        for name, count in per_file.items():
            log.info(f"  {name}: {count} чанк.")
        log.info(f"Документов: {len(documents)}, чанков: {len(chunks)}")

        # Пакетная индексация: виден прогресс, меньше риск таймаута
        bs = self.cfg.embed_batch_size
        store: Optional[FAISS] = None
        for start in range(0, len(chunks), bs):
            batch = chunks[start:start + bs]
            if store is None:
                store = FAISS.from_documents(batch, self.embeddings)
            else:
                store.add_documents(batch)
            log.info(f"  embeddings: {min(start + bs, len(chunks))}/{len(chunks)}")

        self.vector_store = store
        log.info("✓ Векторное хранилище создано!")

    def save_vector_store(self, path: str) -> None:
        if self.vector_store is None:
            log.error("Нет векторного хранилища для сохранения!")
            return
        Path(path).mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(path), index_name=INDEX_NAME)
        log.info(f"✓ Хранилище сохранено в {path}")

    def load_vector_store(self, path: str) -> bool:
        try:
            self.vector_store = FAISS.load_local(
                str(path),
                self.embeddings,
                index_name=INDEX_NAME,
                allow_dangerous_deserialization=True,
            )
            log.info(f"✓ Хранилище загружено из {path}")
            return True
        except Exception as e:
            log.error(f"✗ Ошибка при загрузке хранилища: {e}")
            return False

    # ---------- Актуальность индекса ----------
    def docs_signature(
        self, docs_dir: str, supported_formats: tuple[str, ...] = SUPPORTED_FORMATS
    ) -> dict:
        """Отпечаток: какие файлы (размер + время изменения) и с какими настройками проиндексированы."""
        directory = Path(docs_dir)
        files: Dict[str, list] = {}
        if directory.exists():
            for path in sorted(directory.rglob("*")):
                if self._is_indexable(path, supported_formats):
                    st = path.stat()
                    files[path.relative_to(directory).as_posix()] = [st.st_size, int(st.st_mtime)]
        return {
            "format_version": INDEX_FORMAT_VERSION,
            "embedding_model": self.cfg.embedding_model,
            "chunk_size": self.cfg.chunk_size,
            "chunk_overlap": self.cfg.chunk_overlap,
            "min_chunk_chars": self.cfg.min_chunk_chars,
            "files": files,
        }

    @staticmethod
    def manifest_matches(store_path: str, signature: dict) -> bool:
        manifest = Path(store_path) / MANIFEST_NAME
        try:
            saved = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False  # манифеста нет (старый индекс) — безопаснее пересобрать
        return saved == signature

    @staticmethod
    def save_manifest(store_path: str, signature: dict) -> None:
        Path(store_path).mkdir(parents=True, exist_ok=True)
        (Path(store_path) / MANIFEST_NAME).write_text(
            json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # ---------- Подготовка к вопросам ----------
    def create_qa_chain(self) -> None:
        """Строит матрицу векторов, список файлов и подгружает кэш сводок."""
        vs = self.vector_store
        if vs is None:
            log.error("Сначала загрузите документы или хранилище!")
            return

        n = vs.index.ntotal
        mat = np.asarray(vs.index.reconstruct_n(0, n), dtype="float32")
        mat /= np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-9)
        self.mat = mat
        self.row_docs = [vs.docstore._dict[vs.index_to_docstore_id[i]] for i in range(n)]  # type: ignore[attr-defined]

        rows: Dict[str, List[int]] = {}
        for i, d in enumerate(self.row_docs):
            name = d.metadata.get("filename")
            if name:
                rows.setdefault(name, []).append(i)
                if name not in self.file_paths and d.metadata.get("source"):
                    self.file_paths[name] = Path(d.metadata["source"])
        self.file_rows = {name: np.asarray(r) for name, r in rows.items()}
        self.known_files = {name.lower(): name for name in rows}

        listed = sorted(rows)
        shown = listed[: self.cfg.max_listed_files]
        self.files_line = ", ".join(shown) + (
            f" … (ещё {len(listed) - len(shown)})" if len(listed) > len(shown) else ""
        )

        self._load_summaries()
        ready = sum(1 for f in rows if self.get_summary(f, create=False))
        log.info(
            f"✓ Готово! Файлов в индексе: {len(rows)}, сводок готово: {ready}/{len(rows)}"
            + ("  (/summarize — подготовить остальные)" if ready < len(rows) else "")
        )

    # ---------- Поиск ----------
    def _embed(self, text: str) -> Tuple[List[float], np.ndarray]:
        """Один запрос к Ollama на вопрос: сырой вектор для FAISS и нормированный для numpy."""
        raw = self.embeddings.embed_query(text)
        unit = np.asarray(raw, dtype="float32")
        unit /= max(float(np.linalg.norm(unit)), 1e-9)
        return raw, unit

    def _search(self, raw_vec: List[float]) -> List[Document]:
        vs = self.vector_store
        assert vs is not None
        if self.cfg.use_mmr:  # MMR берёт разнообразные фрагменты — контекст богаче, меньше повторов
            return vs.max_marginal_relevance_search_by_vector(
                raw_vec, k=self.cfg.top_k, fetch_k=self.cfg.fetch_k, lambda_mult=0.6
            )
        return vs.similarity_search_by_vector(raw_vec, k=self.cfg.top_k)

    def _top_chunks(self, qunit: np.ndarray, files: List[str], k: int) -> List[Document]:
        """k самых подходящих к запросу фрагментов из каждого файла (по убыванию релевантности)."""
        assert self.mat is not None
        out: List[Document] = []
        for f in files:
            rows = self.file_rows.get(f)
            if rows is None:
                continue
            sims = self.mat[rows] @ qunit
            out.extend(self.row_docs[rows[i]] for i in np.argsort(-sims)[:k])
        return out

    def _rank_files(self, qunit: np.ndarray, limit: int) -> List[str]:
        """Статьи, ближайшие к запросу: средняя близость трёх лучших фрагментов статьи."""
        assert self.mat is not None
        sims = self.mat @ qunit
        scores = {
            f: float(np.sort(sims[rows])[::-1][:3].mean()) for f, rows in self.file_rows.items()
        }
        return sorted(scores, key=scores.__getitem__, reverse=True)[:limit]

    # ---------- Работа с именами файлов ----------
    def list_files(self) -> List[str]:
        return sorted(self.known_files.values())

    def resolve_file(self, name: str) -> Optional[str]:
        """Находит настоящее имя файла по имени/части имени/имени без расширения."""
        name = unquote(name.strip().strip("`'\"«»")).strip()
        if not name:
            return None
        low = Path(name).name.lower()
        if low in self.known_files:
            return self.known_files[low]
        stem = Path(low).stem
        for k, real in self.known_files.items():
            if Path(k).stem == stem:
                return real
        partial = [real for k, real in self.known_files.items() if low in k]
        return partial[0] if len(partial) == 1 else None

    def pin_file(self, name: str) -> Optional[str]:
        real = self.resolve_file(name)
        if real and real not in self.pinned_files:
            self.pinned_files.append(real)
        return real

    def unpin_files(self) -> None:
        self.pinned_files.clear()

    @staticmethod
    def _linked_names(text: str) -> List[str]:
        names = [Path(unquote(m)).name for m in _MD_LINK_RE.findall(text)]
        names += [Path(unquote(m.strip())).name + ".md" for m in _WIKI_LINK_RE.findall(text)]
        return names

    @staticmethod
    def _key(d: Document) -> tuple:
        return (d.metadata.get("filename"), d.metadata.get("chunk_id"))

    # ---------- Сводки статей ----------
    def _load_summaries(self) -> None:
        try:
            data = json.loads((self.store_dir / SUMMARIES_NAME).read_text(encoding="utf-8"))
            self.summaries = data["items"] if data.get("version") == SUMMARY_VERSION else {}
        except (OSError, ValueError, KeyError):
            self.summaries = {}

    def _save_summaries(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.store_dir / (SUMMARIES_NAME + ".tmp")
        tmp.write_text(
            json.dumps({"version": SUMMARY_VERSION, "items": self.summaries}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
        tmp.replace(self.store_dir / SUMMARIES_NAME)  # атомарно: прерывание не испортит кэш

    def _fingerprint(self, filename: str) -> Optional[list]:
        path = self.file_paths.get(filename)
        try:
            st = path.stat() if path else None
        except OSError:
            return None
        return [st.st_size, int(st.st_mtime)] if st else None

    def get_summary(self, filename: str, create: bool = True) -> str:
        """Сводка статьи из кэша; при create=True и отсутствии — генерируется и сохраняется."""
        fp = self._fingerprint(filename)
        item = self.summaries.get(filename)
        if item and (fp is None or item.get("fp") == fp):
            return item["text"]
        if not create or fp is None:
            return ""

        text = self.load_document(self.file_paths[filename])
        limit = min(self.cfg.summary_input_chars, self._char_budget(self.cfg.summary_max_tokens) - len(SUMMARY_PROMPT) - 200)
        prompt = SUMMARY_PROMPT.format(text=build_summary_input(text, filename, max(limit, 2000)))
        summary = self.llm.generate(prompt, num_predict=self.cfg.summary_max_tokens, temperature=0.2)
        if summary:
            self.summaries[filename] = {"fp": fp, "text": summary}
            self._save_summaries()
        return summary

    def summarize_all(self, files: Optional[List[str]] = None) -> int:
        """Готовит недостающие сводки. Можно прервать Ctrl+C — готовое сохранено."""
        files = files or self.list_files()
        todo = [f for f in files if not self.get_summary(f, create=False)]
        log.info(f"Сводок к подготовке: {len(todo)} из {len(files)}")
        done = 0
        started = time.time()
        try:
            for i, f in enumerate(todo, 1):
                t0 = time.time()
                if self.get_summary(f):
                    done += 1
                log.info(f"  [{i}/{len(todo)}] {f} ({time.time() - t0:.0f} с)")
        except KeyboardInterrupt:
            log.info("Прервано — готовые сводки сохранены.")
        log.info(f"✓ Готово сводок: {done}, всего времени {time.time() - started:.0f} с")
        return done

    def _summary_block(self, files: List[str]) -> str:
        parts = []
        for f in files:
            s = self.get_summary(f, create=False)
            if s:
                parts.append(f"Краткая сводка статьи {f}:\n{s}")
        return "\n\n".join(parts) + "\n\n" if parts else ""

    # ---------- Расширение и фокус ----------
    def _expand_linked(self, docs: List[Document], qunit: np.ndarray) -> List[Document]:
        """Если найденные фрагменты ссылаются на другие файлы индекса — добавляем их."""
        if not (self.cfg.follow_links and self.known_files and self.mat is not None):
            return docs

        have = {d.metadata.get("filename") for d in docs}
        wanted: List[str] = []
        for d in docs:
            for name in self._linked_names(d.page_content):
                real = self.known_files.get(name.lower())
                if real and real not in have and real not in wanted:
                    wanted.append(real)

        extra: List[Document] = []
        for real in wanted[: self.cfg.max_linked_files]:
            hits = self._top_chunks(qunit, [real], 1)
            if hits:
                log.info(f"  ↪ подтянут связанный файл: {real}")
                extra.extend(hits)
        return docs + extra

    def _detect_focus(self, question: str, docs: List[Document]) -> List[str]:
        """Какие файлы читать «глубоко»: закреплённые → упомянутые в вопросе → файл лучшего совпадения."""
        cap = self.cfg.max_focus_files
        if self.pinned_files:
            return self.pinned_files[:cap]

        qlow = question.lower()
        mentioned: List[str] = []
        for low, real in self.known_files.items():
            stem = Path(low).stem
            if low in qlow or (len(stem) >= 4 and stem in qlow):
                mentioned.append(real)
        if mentioned:
            return mentioned[:cap]

        if not docs:
            return []
        top = docs[0].metadata.get("filename")
        if not top:
            return []
        focus = [top]
        top_stem = Path(top).stem.lower()
        for low, real in self.known_files.items():  # «соседи»: article.md рядом с article.xml
            if real != top and Path(low).stem == top_stem:
                focus.append(real)
        return focus[:cap]

    def _char_budget(self, reserve_tokens: Optional[int] = None) -> int:
        """Сколько символов промпта влезает в окно, если оставить reserve_tokens под ответ."""
        if reserve_tokens is None:
            reserve_tokens = (
                self.cfg.num_predict if self.cfg.num_predict > 0 else self.cfg.answer_reserve_tokens
            )
        return int((self.cfg.num_ctx - reserve_tokens) * self.cfg.chars_per_token)

    def _build_context(
        self,
        focus_docs: List[Document],
        other_docs: List[Document],
        question: str,
        history: str,
        overhead_extra: int = 0,
    ) -> tuple[str, List[Document]]:
        """
        Собирает контекст так, чтобы весь промпт гарантированно влез в num_ctx.
        Фокусные файлы получают приоритетную долю бюджета, их фрагменты идут в порядке текста.
        Возвращает текст и список фрагментов, которые модель реально увидит.
        """
        overhead = len(PROMPT.template) + len(history) + len(question) + overhead_extra
        budget = max(self._char_budget() - overhead, 1500)
        focus_budget = int(budget * self.cfg.focus_share) if other_docs else budget

        def cost(d: Document) -> int:
            return len(d.page_content) + len(d.metadata.get("filename", "?")) + 24

        # 1) фокусные: отбираем по релевантности, потом раскладываем в порядке документа
        chosen_focus: List[Document] = []
        size = 0
        for d in focus_docs:
            c = cost(d)
            if size + c > focus_budget:
                if not chosen_focus:
                    chosen_focus.append(d)
                    size = min(c, focus_budget)
                continue
            chosen_focus.append(d)
            size += c
        file_order: Dict[str, int] = {}
        for d in focus_docs:
            file_order.setdefault(d.metadata.get("filename", "?"), len(file_order))
        chosen_focus.sort(
            key=lambda d: (file_order[d.metadata.get("filename", "?")], d.metadata.get("chunk_id", 0))
        )

        # 2) остальное — в оставшееся место
        chosen_other: List[Document] = []
        for d in other_docs:
            c = cost(d)
            if size + c > budget:
                continue
            chosen_other.append(d)
            size += c

        used = chosen_focus + chosen_other
        if not used and other_docs:
            used = [other_docs[0]]  # хотя бы самый релевантный фрагмент отдаём (урезанным)

        blocks = [
            f"[Источник {i}: {d.metadata.get('filename', '?')}]\n{d.page_content}"[:budget]
            for i, d in enumerate(used, 1)
        ]

        if len(chosen_focus) < len(focus_docs):
            log.info(
                f"  (из фокусного файла в окно вошло {len(chosen_focus)} из {len(focus_docs)} фрагментов; "
                "остальное компенсирует сводка статьи. Для большего охвата увеличьте --num-ctx)"
            )
        composition = ", ".join(
            f"{name} ×{n}" for name, n in Counter(d.metadata.get("filename", "?") for d in used).items()
        )
        log.info(f"\n  контекст: {composition}")
        return "\n\n".join(blocks), used

    def _format_history(self) -> str:
        if not self.history:
            return "(пусто)"
        limit = self.cfg.history_answer_chars
        turns = self.history[-self.cfg.max_history_turns:]
        return "\n".join(
            f"Вопрос: {q}\nОтвет: {a[:limit] + ('…' if len(a) > limit else '')}"
            for q, a in turns
        )

    # ---------- Генерация ----------
    def _generate(
        self,
        question: str,
        docs: List[Document],
        focus_files: List[str],
        qunit: np.ndarray,
        allow_read: bool,
        on_token: Optional[Callable[[str], None]],
    ) -> Tuple[str, List[Document], Optional[Tuple[str, str]]]:
        """
        Один проход: собрать контекст и получить ответ потоком.
        Если модель вместо ответа просит файл ([[ЧИТАТЬ: ...]]), возвращает (имя, что искать).
        """
        focus_docs = self._top_chunks(qunit, focus_files, max(3, self.cfg.focus_chunks // max(len(focus_files), 1)))
        focus_keys = {self._key(d) for d in focus_docs}
        others = [d for d in docs if self._key(d) not in focus_keys]

        history = self._format_history()
        allow_read = allow_read and bool(self.known_files)
        files_line = f"Доступные файлы: {self.files_line}\n\n" if allow_read else ""
        files_line += self._summary_block(focus_files)
        read_rule = READ_RULE if allow_read else ""

        context, used = self._build_context(
            focus_docs, others, question, history, len(files_line) + len(read_rule)
        )
        prompt = PROMPT.format(
            history=history,
            context=context,
            question=question,
            files=files_line,
            read_rule=read_rule,
        )

        # Первые символы буферизуем: если ответ начинается с «[[», это может быть запрос файла,
        # и пользователю его показывать не нужно.
        parts: List[str] = []
        head = ""
        mode = ""  # "" — ещё не ясно, "text" — обычный ответ, "maybe_read" — похоже на запрос файла
        for token in self.llm.stream(prompt):
            parts.append(token)
            if mode == "":
                head += token
                if len(head.lstrip()) < 2:
                    continue
                mode = "maybe_read" if (allow_read and head.lstrip().startswith("[[")) else "text"
                if mode == "text" and on_token:
                    on_token(head)
                continue
            if mode == "text" and on_token:
                on_token(token)
        answer = "".join(parts).strip()

        if mode == "":  # совсем короткий ответ
            if on_token and head:
                on_token(head)
        elif mode == "maybe_read":
            m = _READ_RE.match(answer) if len(answer) < 400 else None
            if m:
                return answer, used, (m.group(1), (m.group(2) or "").strip())
            if on_token:  # это был обычный ответ, начинающийся с «[[» — выводим целиком
                on_token(answer)
        return answer, used, None

    def query(
        self, question: str, on_token: Optional[Callable[[str], None]] = None
    ) -> Dict:
        """
        Отвечает на вопрос. Если передан on_token, ответ выводится потоком —
        длинный ответ не нужно ждать целиком.
        """
        if self.vector_store is None or self.mat is None:
            log.error("Сначала создайте QA-цепь!")
            return {}

        try:
            raw, qunit = self._embed(question)  # единственный embedding-запрос на вопрос
            docs = self._search(raw)
            docs = self._expand_linked(docs, qunit)
            focus = self._detect_focus(question, docs)

            answer, used, request = self._generate(
                question, docs, focus, qunit, self.cfg.allow_file_requests, on_token
            )
            if request:  # модель попросила прочитать конкретный файл
                name, what = request
                real = self.resolve_file(name)
                if real:
                    log.info(f"  ↪ модель запросила файл: {real}" + (f" ({what})" if what else ""))
                    wunit = self._embed(what)[1] if what else qunit
                    answer, used, _ = self._generate(question, docs, [real], wunit, False, on_token)
                else:
                    log.info(f"  ↪ модель запросила неизвестный файл «{name}», отвечаю по найденному")
                    answer, used, _ = self._generate(question, docs, focus, qunit, False, on_token)
        except Exception as e:
            log.error(f"✗ Ошибка при получении ответа: {e}")
            return {}

        self.history.append((question, answer))
        # В источниках — только то, что модель действительно получила в промпте
        sources = list(dict.fromkeys(d.metadata.get("source", "?") for d in used))
        return {"answer": answer, "sources": sources}

    # ---------- Зонтичный обзор ----------
    def _select_review_files(self, topic: str) -> List[str]:
        if not topic or topic.lower() in {"all", "все", "всё", "*"}:
            return self.list_files()
        _, qunit = self._embed(topic)
        return self._rank_files(qunit, self.cfg.review_max_articles)

    @staticmethod
    def _pack(items: List[str], budget: int) -> List[List[str]]:
        """Раскладывает карточки по группам, каждая из которых помещается в budget символов."""
        batches: List[List[str]] = []
        cur: List[str] = []
        size = 0
        for it in items:
            if cur and size + len(it) > budget:
                batches.append(cur)
                cur, size = [], 0
            cur.append(it)
            size += len(it) + 2
        if cur:
            batches.append(cur)
        return batches

    def review(
        self, topic: str = "", on_token: Optional[Callable[[str], None]] = None
    ) -> Dict:
        """
        Зонтичный обзор: статьи по теме → карточки → (при нехватке окна — сжатие группами) → обзор.
        """
        if self.mat is None:
            log.error("Сначала создайте QA-цепь!")
            return {}
        try:
            files = self._select_review_files(topic)
            if not files:
                log.error("Нет статей для обзора.")
                return {}
            log.info(f"Статей в обзоре: {len(files)}")

            cards: List[Tuple[int, str, str]] = []
            missing = [f for f in files if not self.get_summary(f, create=False)]
            if missing:
                log.info(f"Готовлю недостающие сводки: {len(missing)} (один раз, дальше берутся из кэша)")
            for f in files:
                if f in missing:
                    log.info(f"  сводка: {f}")
                s = self.get_summary(f)
                if s:
                    cards.append((len(cards) + 1, f, s))
            if not cards:
                log.error("Не удалось подготовить ни одной сводки.")
                return {}

            topic_label = topic or "все статьи коллекции"
            items = [f"[{n}] {f}\n{s}" for n, f, s in cards]
            budget = self._char_budget(self.cfg.review_answer_tokens) - len(REVIEW_PROMPT) - len(topic_label)

            # reduce: пока карточки не помещаются в окно — сжимаем их группами в заметки
            for level in range(1, 5):
                batches = self._pack(items, budget)
                if len(batches) == 1:
                    break
                if len(batches) >= len(items):  # сжатие не помогает (окно слишком мало)
                    log.warning("⚠ Окно слишком мало для всех карточек — часть статей не попадёт в обзор. Увеличьте --num-ctx.")
                    batches = batches[:1]
                    break
                log.info(f"  сжатие, этап {level}: {len(items)} → {len(batches)} групп")
                items = [
                    self.llm.generate(
                        MAP_PROMPT.format(topic=topic_label, notes="\n\n".join(b)),
                        num_predict=self.cfg.review_map_tokens,
                        temperature=0.2,
                    )
                    for b in batches
                ]
            else:
                batches = self._pack(items, budget)[:1]

            prompt = REVIEW_PROMPT.format(topic=topic_label, notes="\n\n".join(batches[0]))
            parts: List[str] = []
            for token in self.llm.stream(
                prompt, num_predict=self.cfg.review_answer_tokens, temperature=0.3
            ):
                parts.append(token)
                if on_token:
                    on_token(token)
        except Exception as e:
            log.error(f"✗ Ошибка при подготовке обзора: {e}")
            return {}

        return {
            "answer": "".join(parts).strip(),
            "sources": [f"[{n}] {f}" for n, f, _ in cards],
            "topic": topic_label,
        }

    def clear_history(self) -> None:
        self.history.clear()


# ──────────────────────────────── CLI ─────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RAG на Ollama + FAISS")
    p.add_argument("--docs", default=str(DOCS_DIR), help="папка с документами")
    p.add_argument("--store", default=str(DOCS_DIR), help="папка индекса FAISS и сводок")
    p.add_argument("--rebuild", action="store_true", help="пересоздать индекс")
    p.add_argument("--model", default=RAGConfig.model_name)
    p.add_argument("--embedding-model", default=RAGConfig.embedding_model)
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--num-ctx", type=int, default=RAGConfig.num_ctx,
                   help="окно контекста в токенах (промпт + ответ); для длинных статей 16384+")
    p.add_argument("--num-predict", type=int, default=RAGConfig.num_predict,
                   help="макс. токенов ответа, -1 = без лимита")
    p.add_argument("-k", "--top-k", type=int, default=RAGConfig.top_k)
    p.add_argument("--focus-chunks", type=int, default=RAGConfig.focus_chunks,
                   help="сколько фрагментов брать из фокусного файла")
    p.add_argument("--max-articles", type=int, default=RAGConfig.review_max_articles,
                   help="сколько статей брать в зонтичный обзор по теме")
    p.add_argument("--think", action="store_true", help="включить режим размышлений модели")
    p.add_argument("--no-follow-links", action="store_true",
                   help="не подтягивать файлы, на которые ссылаются найденные фрагменты")
    p.add_argument("--no-file-requests", action="store_true",
                   help="запретить модели самой запрашивать файл через [[ЧИТАТЬ: ...]]")
    return p.parse_args()


def save_review(result: Dict) -> Path:
    """Сохраняет обзор в <корень проекта>/reviews (вне папки документов, чтобы не попасть в индекс)."""
    REVIEWS_DIR.mkdir(parents=True, exist_ok=True)
    path = REVIEWS_DIR / f"review_{datetime.now():%Y%m%d_%H%M%S}.md"
    legend = "\n".join(f"- {s}" for s in result["sources"])
    path.write_text(
        f"# Зонтичный обзор: {result['topic']}\n\n{result['answer']}\n\n## Статьи\n{legend}\n",
        encoding="utf-8",
    )
    return path


def main() -> None:
    args = parse_args()
    config = RAGConfig(
        model_name=args.model,
        embedding_model=args.embedding_model,
        ollama_host=args.host,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        top_k=args.top_k,
        focus_chunks=args.focus_chunks,
        review_max_articles=args.max_articles,
        think=args.think,
        follow_links=not args.no_follow_links,
        allow_file_requests=not args.no_file_requests,
    )

    try:
        rag = RAGSystem(config, store_dir=args.store)
    except ConnectionError as e:
        log.error(f"\n{'=' * 60}\nОШИБКА: {e}\n{'=' * 60}")
        sys.exit(1)

    signature = rag.docs_signature(args.docs)
    index_file = Path(args.store) / f"{INDEX_NAME}.faiss"
    up_to_date = RAGSystem.manifest_matches(args.store, signature)
    if index_file.exists() and not args.rebuild and not up_to_date:
        log.info("Документы или настройки индексации изменились — пересоздаю индекс...")

    loaded = (
        index_file.exists()
        and not args.rebuild
        and up_to_date
        and rag.load_vector_store(args.store)
    )
    if not loaded:
        documents = rag.load_documents_from_directory(args.docs)
        if not documents:
            log.error("Не удалось загрузить документы!")
            sys.exit(1)
        rag.create_vector_store(documents)
        if rag.vector_store is None:
            sys.exit(1)
        rag.save_vector_store(args.store)
        RAGSystem.save_manifest(args.store, signature)

    rag.create_qa_chain()

    print("\n" + "=" * 60)
    print("RAG система готова. 'exit' — выход, '/clear' — сбросить историю")
    print("'/files' — список файлов, '/file <имя>' — читать файл целиком, '/unfile' — снять")
    print("'/summarize' — сводки статей, '/review [тема]' — зонтичный обзор")
    print("=" * 60 + "\n")

    def show_result(result: Dict, title: str) -> None:
        if not result:
            return
        print(f"\n\n{title}:")
        for src in result["sources"]:
            print(f"  - {src}")
        print()

    while True:
        try:
            question = input("Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.lower() in {"exit", "quit"}:
            break
        if question == "/clear":
            rag.clear_history()
            print("История очищена.\n")
            continue
        if question == "/files":
            for name in rag.list_files():
                mark = " 📌" if name in rag.pinned_files else ""
                done = "✓" if rag.get_summary(name, create=False) else " "
                print(f"  [{done}] {name}{mark}")
            print("  ([✓] — сводка готова)\n")
            continue
        if question.startswith("/file"):
            arg = question[len("/file"):].strip()
            if not arg:
                print("Использование: /file <имя файла>\n")
                continue
            real = rag.pin_file(arg)
            print(f"Закреплён: {real}\n" if real else f"Файл «{arg}» не найден (см. /files).\n")
            continue
        if question == "/unfile":
            rag.unpin_files()
            print("Закрепление снято.\n")
            continue
        if question.startswith("/summarize"):
            arg = question[len("/summarize"):].strip()
            try:
                if not arg:
                    rag.summarize_all()
                else:
                    real = rag.resolve_file(arg)
                    print(rag.get_summary(real) if real else f"Файл «{arg}» не найден (см. /files).")
            except KeyboardInterrupt:
                print("\nПрервано.")
            print()
            continue
        if not question:
            continue

        is_review = question.startswith("/review") or is_overview_question(question)
        try:
            if is_review:
                topic = question[len("/review"):].strip() if question.startswith("/review") else question
                print("\nОбзор: ", end="", flush=True)
                result = rag.review(topic, on_token=lambda t: print(t, end="", flush=True))
                show_result(result, "Статьи в обзоре")
                if result:
                    print(f"Обзор сохранён: {save_review(result)}\n")
            else:
                print("\nОтвет: ", end="", flush=True)
                result = rag.query(question, on_token=lambda t: print(t, end="", flush=True))
                show_result(result, "Источники")
        except KeyboardInterrupt:
            print("\nПрервано.\n")


if __name__ == "__main__":
    main()