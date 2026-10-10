#!/usr/bin/env python3
"""
Загрузчик статей из PubMed + полные тексты из PubMed Central (PMC).

ВАЖНО: PubMed хранит только метаданные и аннотации (abstract). Полный текст
доступен через PMC — это та часть статей, которая там размещена (в основном
open access). Скрипт сам находит PMCID у каждой статьи и докачивает полный текст.

Примеры запуска:
    python pubmed_downloader.py
    python pubmed_downloader.py -q "expert systems medicine" -q "clinical decision support"
    python pubmed_downloader.py --fulltext-formats xml,md,txt,json
    python pubmed_downloader.py --formats xml,json --year-from 2018 --limit 100
    python pubmed_downloader.py --no-only-full-text      # все статьи PubMed, полный текст где доступен
    python pubmed_downloader.py --no-clean               # не удалять папку перед загрузкой

По умолчанию папка вывода ОЧИЩАЕТСЯ перед загрузкой, а поиск идёт прямо в PMC
среди open-access статей — у них гарантированно есть полный текст.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ============================================
# ⚙️ НАСТРОЙКИ ПО УМОЛЧАНИЮ (можно менять здесь)
# ============================================
DEFAULT_OUTPUT_DIR = r"C:\Users\admin\PyCharmMiscProject\documents"
DEFAULT_EMAIL = "zidzidovich@gmail.com"
DEFAULT_QUERIES = [
    "expert systems medicine",
    "clinical decision support systems",
    "artificial intelligence diagnosis",
    "machine learning medical",
    "intelligent diagnostic",
]

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
ABSTRACT_FORMATS = ("xml", "medline", "json", "txt", "md")
FULLTEXT_FORMATS = ("xml", "json", "txt", "md")

ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
ET.register_namespace("mml", "http://www.w3.org/1998/Math/MathML")


# ============================================
# 🧰 Вспомогательные функции
# ============================================
def text_of(elem) -> str:
    """Весь текст элемента (включая вложенные теги <i>, <sub> и т.д.)."""
    if elem is None:
        return ""
    return re.sub(r"\s+", " ", "".join(elem.itertext())).strip()


def safe_name(name: str, max_len: int = 80) -> str:
    """Имя файла, безопасное для Windows/Linux/macOS."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:max_len].rstrip(" .") or "article"


def elem_to_xml(elem) -> str:
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(elem, encoding="unicode")


def normalize_pmcid(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    return f"PMC{digits}" if digits else ""


# ============================================
# 📄 Разбор PubMed XML (метаданные + abstract)
# ============================================
def parse_pubmed_article(node) -> dict | None:
    citation = node.find("MedlineCitation")
    art = citation.find("Article") if citation is not None else None
    if art is None:
        return None

    abstract_parts = []
    for at in art.findall("Abstract/AbstractText"):
        label, txt = at.get("Label"), text_of(at)
        if txt:
            abstract_parts.append(f"{label}: {txt}" if label else txt)

    authors = []
    for a in art.findall("AuthorList/Author"):
        collective = text_of(a.find("CollectiveName"))
        if collective:
            authors.append(collective)
            continue
        first = text_of(a.find("ForeName")) or text_of(a.find("Initials"))
        name = " ".join(x for x in (first, text_of(a.find("LastName"))) if x)
        if name:
            authors.append(name)

    pub_date = art.find("Journal/JournalIssue/PubDate")
    year = text_of(pub_date.find("Year")) if pub_date is not None else ""
    if not year and pub_date is not None:
        m = re.search(r"\d{4}", text_of(pub_date.find("MedlineDate")))
        year = m.group(0) if m else ""

    ids = {a.get("IdType"): text_of(a) for a in node.findall("PubmedData/ArticleIdList/ArticleId")}
    doi = ids.get("doi", "")
    if not doi:
        for e in art.findall("ELocationID"):
            if e.get("EIdType") == "doi":
                doi = text_of(e)

    return {
        "pmid": text_of(citation.find("PMID")),
        "pmcid": normalize_pmcid(ids.get("pmc", "")),
        "doi": doi,
        "title": text_of(art.find("ArticleTitle")),
        "abstract": "\n\n".join(abstract_parts),
        "authors": authors,
        "journal": text_of(art.find("Journal/Title")),
        "year": year,
        "keywords": [text_of(k) for k in citation.findall(".//KeywordList/Keyword") if text_of(k)],
        "mesh_terms": [text_of(m) for m in citation.findall(".//MeshHeading/DescriptorName")],
    }


# ============================================
# 📚 Разбор PMC XML (JATS) — полный текст
# ============================================
def _jats_blocks(node, level: int, out: list) -> None:
    """Рекурсивно превращает тело статьи (JATS) в Markdown-блоки."""
    for child in node:
        tag = child.tag
        if tag == "sec":
            title = child.find("title")
            if title is not None:
                out.append("#" * min(level, 6) + " " + text_of(title))
            _jats_blocks(child, level + 1, out)
        elif tag == "p":
            txt = text_of(child)
            if txt:
                out.append(txt)
        elif tag == "list":
            for li in child.findall("list-item"):
                out.append("- " + text_of(li))
        elif tag in ("fig", "table-wrap"):
            cap = text_of(child.find("caption"))
            if cap:
                out.append(f"*[{'Figure' if tag == 'fig' else 'Table'}] {cap}*")
        elif tag in ("boxed-text", "disp-quote"):
            _jats_blocks(child, level + 1, out)


def _abstract_text(abs_elem) -> str:
    parts = []
    for child in abs_elem:
        if child.tag == "sec":
            title = text_of(child.find("title"))
            para = " ".join(t for t in (text_of(x) for x in child.findall("p")) if t)
            if title and para:
                parts.append(f"{title}: {para}")
            elif para or title:
                parts.append(para or title)
        elif child.tag == "p":
            t = text_of(child)
            if t:
                parts.append(t)
    return "\n\n".join(parts)


def parse_pmc_article(art) -> dict:
    """Метаданные + полный текст из JATS XML статьи PMC."""
    meta = art.find("front/article-meta")
    if meta is None:
        meta = ET.Element("article-meta")
    ids = {a.get("pub-id-type"): text_of(a) for a in meta.findall("article-id")}

    abstracts = meta.findall("abstract")
    main_abs = next((a for a in abstracts if a.get("abstract-type") is None),
                    abstracts[0] if abstracts else None)

    authors = []
    for c in meta.findall("contrib-group/contrib"):
        if c.get("contrib-type") != "author":
            continue
        collab = text_of(c.find("collab"))
        if collab:
            authors.append(collab)
            continue
        name = " ".join(x for x in (text_of(c.find("name/given-names")),
                                    text_of(c.find("name/surname"))) if x)
        if name:
            authors.append(name)

    year = ""
    for pub in meta.findall("pub-date"):
        year = text_of(pub.find("year"))
        if year:
            break

    body = art.find("body")
    blocks: list[str] = []
    if body is not None:
        _jats_blocks(body, 2, blocks)
    full_text = "\n\n".join(blocks)

    return {
        "pmid": ids.get("pmid", ""),
        "pmcid": normalize_pmcid(ids.get("pmcid") or ids.get("pmc") or ""),
        "doi": ids.get("doi", ""),
        "title": text_of(meta.find("title-group/article-title")),
        "abstract": _abstract_text(main_abs) if main_abs is not None else "",
        "authors": authors,
        "journal": text_of(art.find("front/journal-meta//journal-title")),
        "year": year,
        "keywords": [t for t in (text_of(k) for k in meta.findall(".//kwd-group/kwd")) if t],
        "mesh_terms": [],
        "full_text": full_text,
        "has_body": len(full_text) > 200,
    }


# ============================================
# 🖨️ Рендер в txt / md / json
# ============================================
def render_text(rec: dict, md: bool, with_full_text: bool = False) -> str:
    out = [f"# {rec['title']}" if md else rec["title"], ""]
    meta = (
        ("Authors", ", ".join(rec["authors"])),
        ("Journal", rec["journal"]),
        ("Year", rec["year"]),
        ("PMID", rec["pmid"]),
        ("PMCID", rec["pmcid"]),
        ("DOI", rec["doi"]),
        ("Keywords", "; ".join(rec["keywords"])),
    )
    for label, value in meta:
        if value:
            out.append(f"- **{label}:** {value}" if md else f"{label}: {value}")
    if rec["abstract"]:
        out += ["", "## Abstract" if md else "ABSTRACT", "", rec["abstract"]]
    if with_full_text and rec.get("full_text"):
        body = rec["full_text"]
        if not md:
            body = re.sub(r"(?m)^#+\s*", "", body)
        out += ["", "## Full text" if md else "FULL TEXT", "", body]
    return "\n".join(out) + "\n"


# ============================================
# 🌐 Основной класс
# ============================================
class ArticleDownloader:
    def __init__(self, output_dir, email, api_key=None, clean=True):
        self.output_dir = Path(output_dir)
        self.email = email
        self.api_key = api_key
        # Лимиты NCBI: 3 запроса/сек без ключа, 10 — с ключом
        self.min_interval = 0.11 if api_key else 0.35
        self._last_call = 0.0

        self.session = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers["User-Agent"] = f"ArticleDownloader/2.0 (Research Tool; Contact: {email})"

        if clean and self.output_dir.exists():
            resolved = self.output_dir.resolve()
            # защита от случайного удаления корня диска или домашней папки
            if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
                sys.exit(f"⛔ Отказ: нельзя очищать {resolved}. Укажите отдельную папку для статей.")
            shutil.rmtree(self.output_dir)
            print(f"🗑️ Старая папка очищена: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Папка: {self.output_dir}\n")

    # ---------- сеть ----------
    def _call(self, endpoint: str, params: dict, timeout: int = 60) -> requests.Response:
        params = {**params, "tool": "ArticleDownloader", "email": self.email}
        if self.api_key:
            params["api_key"] = self.api_key
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        resp = self.session.post(EUTILS + endpoint, data=params, timeout=timeout)
        self._last_call = time.monotonic()
        resp.raise_for_status()
        return resp

    # ---------- поиск ----------
    def search(self, query, max_results=30, year_from=None, year_to=None,
               db="pubmed", sort="date") -> list[str]:
        """db='pubmed' -> PMID; db='pmc' -> UID статей PMC (open access, PMCID без префикса)."""
        term = f"({query})"
        params = {"db": db, "retmax": max_results, "retmode": "json"}
        if db == "pmc":
            term += " AND open access[filter]"  # у open-access статей гарантирован полный текст
        elif sort:
            params["sort"] = sort
        params["term"] = term
        if year_from or year_to:
            params.update(datetype="pdat",
                          mindate=str(year_from or 1900),
                          maxdate=str(year_to or datetime.now().year))
        try:
            data = self._call("esearch.fcgi", params, timeout=30).json()["esearchresult"]
            ids = data.get("idlist", [])
            print(f"✓ '{query}' [{db}]: найдено {data.get('count', '?')}, взято {len(ids)}")
            return ids
        except Exception as e:
            print(f"✗ Ошибка поиска '{query}': {e}")
            return []

    # ---------- загрузка PubMed (метаданные + abstract) ----------
    def fetch_pubmed(self, ids, batch_size=100):
        records, raw_xml = {}, {}
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            try:
                resp = self._call("efetch.fcgi", {"db": "pubmed", "id": ",".join(batch),
                                                  "retmode": "xml", "rettype": "xml"})
                root = ET.fromstring(resp.content)
            except Exception as e:
                print(f"✗ Ошибка загрузки PubMed (пакет {i // batch_size + 1}): {e}")
                continue
            for node in root.findall("PubmedArticle"):
                rec = parse_pubmed_article(node)
                if rec and rec["pmid"]:
                    records[rec["pmid"]] = rec
                    raw_xml[rec["pmid"]] = elem_to_xml(node)
            print(f"📥 PubMed: {min(i + batch_size, len(ids))}/{len(ids)}")
        return records, raw_xml

    def fetch_medline(self, ids, batch_size=100) -> dict:
        result = {}
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            try:
                resp = self._call("efetch.fcgi", {"db": "pubmed", "id": ",".join(batch),
                                                  "rettype": "medline", "retmode": "text"})
            except Exception as e:
                print(f"✗ Ошибка загрузки MEDLINE: {e}")
                continue
            text = resp.content.decode("utf-8", errors="replace").strip()
            for chunk in re.split(r"\n\s*\n(?=PMID- )", text):
                m = re.match(r"PMID- (\d+)", chunk)
                if m:
                    result[m.group(1)] = chunk.strip() + "\n"
        return result

    # ---------- загрузка полных текстов из PMC ----------
    def fetch_pmc(self, pmcids, batch_size=20) -> dict:
        """Возвращает {PMCID: (info, xml)} только для статей, где есть полный текст."""
        result = {}
        no_body = 0
        for i in range(0, len(pmcids), batch_size):
            batch = pmcids[i:i + batch_size]
            try:
                resp = self._call("efetch.fcgi", {"db": "pmc",
                                                  "id": ",".join(p[3:] for p in batch),
                                                  "retmode": "xml"}, timeout=120)
                root = ET.fromstring(resp.content)
            except Exception as e:
                print(f"✗ Ошибка загрузки PMC (пакет {i // batch_size + 1}): {e}")
                continue
            articles = [root] if root.tag == "article" else root.findall("article")
            if not articles:
                print(f"⚠️ PMC вернул ответ без статей: {resp.text[:300]!r}")
            for art in articles:
                info = parse_pmc_article(art)
                if info["pmcid"] and info["has_body"]:
                    result[info["pmcid"]] = (info, elem_to_xml(art))
                else:
                    no_body += 1
            print(f"📥 PMC: {min(i + batch_size, len(pmcids))}/{len(pmcids)}")
        if no_body:
            print(f"⚠️ У {no_body} статей PMC не отдал полный текст (не open access или embargo)")
        return result

    # ---------- сохранение ----------
    def _write(self, path: Path, content: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return str(path.relative_to(self.output_dir))

    def run(self, queries, max_results=30, limit=None, formats=(), fulltext_formats=(),
            year_from=None, year_to=None, only_full_text=True, sort="date"):
        use_pmc = bool(only_full_text and fulltext_formats)
        db = "pmc" if use_pmc else "pubmed"
        print("🔧 Режим: " + ("поиск в PMC, только open-access статьи с полным текстом"
                             if use_pmc else "поиск в PubMed, полный текст — где доступен") + "\n")

        # 1. Поиск
        all_ids: list[str] = []
        for q in queries:
            all_ids.extend(self.search(q, max_results, year_from, year_to, db, sort))
        all_ids = list(dict.fromkeys(all_ids))  # уникальные, порядок сохраняется
        if limit:
            all_ids = all_ids[:limit]
        print(f"\n📊 Уникальных статей к загрузке: {len(all_ids)}\n")
        if not all_ids:
            return

        # 2. Загрузка данных
        records, raw_xml, medline, fulltexts = {}, {}, {}, {}
        if use_pmc:
            pmc_ids = [f"PMC{u}" for u in all_ids]
            fulltexts = self.fetch_pmc(pmc_ids)
            for pmcid, (info, _) in fulltexts.items():
                records[pmcid] = {k: v for k, v in info.items() if k not in ("full_text", "has_body")}
            order = [p for p in pmc_ids if p in records]
            pmids = [r["pmid"] for r in records.values() if r["pmid"]]
            if pmids and {"xml", "medline"} & set(formats):
                _, raw_xml = self.fetch_pubmed(pmids)  # оригинальный PubMed XML для файлов _abstract
        else:
            records, raw_xml = self.fetch_pubmed(all_ids)
            order = [p for p in all_ids if p in records]
            pmids = list(records)
            if fulltext_formats:
                pmcids = [r["pmcid"] for r in records.values() if r["pmcid"]]
                print(f"\n🔎 В PMC найдено {len(pmcids)} из {len(records)} статей, загружаю полные тексты...")
                fulltexts = self.fetch_pmc(pmcids)
        if "medline" in formats and pmids:
            medline = self.fetch_medline(pmids)

        if not records:
            print("⚠️ Не удалось получить ни одной статьи — сохранять нечего.")
            return

        # 3. Сохранение
        print("\n💾 Сохранение файлов...")
        index_rows = []
        for key in order:
            rec = records[key]
            ident = rec["pmid"] or rec["pmcid"]
            base = f"{ident}_{safe_name(rec['title'])}"
            files = []

            folder = self.output_dir  # все файлы лежат прямо в папке вывода
            for fmt in formats:
                if fmt == "xml" and rec["pmid"] in raw_xml:
                    files.append(self._write(folder / f"{base}_abstract.xml", raw_xml[rec["pmid"]]))
                elif fmt == "medline" and rec["pmid"] in medline:
                    files.append(self._write(folder / f"{base}_abstract_medline.txt", medline[rec["pmid"]]))
                elif fmt == "json":
                    files.append(self._write(folder / f"{base}_abstract.json",
                                             json.dumps(rec, ensure_ascii=False, indent=2)))
                elif fmt in ("txt", "md"):
                    files.append(self._write(folder / f"{base}_abstract.{fmt}", render_text(rec, fmt == "md")))

            has_full = rec["pmcid"] in fulltexts
            if has_full:
                info, xml = fulltexts[rec["pmcid"]]
                full_rec = {**rec, "full_text": info["full_text"]}
                for fmt in fulltext_formats:
                    if fmt == "xml":
                        files.append(self._write(folder / f"{base}_fulltext.xml", xml))
                    elif fmt == "json":
                        files.append(self._write(folder / f"{base}_fulltext.json",
                                                 json.dumps(full_rec, ensure_ascii=False, indent=2)))
                    elif fmt in ("txt", "md"):
                        files.append(self._write(folder / f"{base}_fulltext.{fmt}",
                                                 render_text(full_rec, fmt == "md", with_full_text=True)))

            index_rows.append({
                "pmid": rec["pmid"], "pmcid": rec["pmcid"], "doi": rec["doi"], "year": rec["year"],
                "journal": rec["journal"], "title": rec["title"],
                "has_full_text": "yes" if has_full else "no", "files": "; ".join(files),
            })

        # 4. Индекс (txt)
        index_path = self.output_dir / "index.txt"
        blocks = []
        for n, r in enumerate(index_rows, 1):
            blocks.append("\n".join([
                f"[{n}] {r['title']}",
                f"PMID: {r['pmid'] or '-'} | PMCID: {r['pmcid'] or '-'} | DOI: {r['doi'] or '-'}",
                f"Journal: {r['journal'] or '-'} | Year: {r['year'] or '-'} | Full text: {r['has_full_text']}",
                f"Files: {r['files']}",
            ]))
        index_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")

        n_full = sum(1 for r in index_rows if r["has_full_text"] == "yes")
        print(f"\n✅ Готово! Статей: {len(index_rows)}, из них с полным текстом: {n_full}")
        if n_full == 0 and fulltext_formats:
            print("⚠️ Полные тексты не скачались. Проверьте сообщения выше (✗ / ⚠️) — "
                  "там указана причина (сеть, ответ PMC).")
        elif not use_pmc and fulltext_formats and n_full < len(index_rows):
            print("ℹ️ Полный текст есть только у статей из PubMed Central (open access). "
                  "Чтобы брать только такие статьи, не используйте --no-only-full-text.")
        print(f"📄 Список статей: {self.output_dir / 'index.txt'}")


# ============================================
# 🚀 Запуск
# ============================================
def parse_formats(value: str, allowed: tuple, label: str) -> list[str]:
    fmts = [f.strip().lower() for f in value.split(",") if f.strip()]
    if "none" in fmts:
        return []
    if "all" in fmts:
        return list(allowed)
    bad = [f for f in fmts if f not in allowed]
    if bad:
        sys.exit(f"Неизвестные форматы для {label}: {', '.join(bad)}. Доступно: {', '.join(allowed)}, all, none")
    return fmts


def parse_args():
    p = argparse.ArgumentParser(description="Загрузка статей из PubMed и полных текстов из PMC")
    p.add_argument("-q", "--query", action="append",
                   help="поисковый запрос (можно указывать несколько раз); по умолчанию — набор из DEFAULT_QUERIES")
    p.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--email", default=DEFAULT_EMAIL)
    p.add_argument("--api-key", default=os.environ.get("NCBI_API_KEY"),
                   help="NCBI API key (ускоряет загрузку до 10 запросов/сек); можно задать через NCBI_API_KEY")
    p.add_argument("--max-results", type=int, default=30, help="сколько статей брать на один запрос")
    p.add_argument("--limit", type=int, default=50, help="общий лимит статей (0 — без лимита)")
    p.add_argument("--formats", default=None,
                   help=f"форматы метаданных/abstract через запятую: {', '.join(ABSTRACT_FORMATS)}, all, none "
                        "(по умолчанию: none в режиме полных текстов, xml в режиме PubMed)")
    p.add_argument("--full-text", action=argparse.BooleanOptionalAction, default=True,
                   help="скачивать полные тексты (по умолчанию включено)")
    p.add_argument("--fulltext-formats", default="xml,md",
                   help=f"форматы полных текстов через запятую: {', '.join(FULLTEXT_FORMATS)}, all")
    p.add_argument("--only-full-text", action=argparse.BooleanOptionalAction, default=True,
                   help="искать прямо в PMC среди open-access статей (полный текст гарантирован). "
                        "--no-only-full-text: искать во всём PubMed, полный текст только где есть")
    p.add_argument("--year-from", type=int)
    p.add_argument("--year-to", type=int)
    p.add_argument("--sort", choices=["date", "relevance"], default="date",
                   help="сортировка (только для поиска в PubMed; в PMC используется релевантность)")
    p.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True,
                   help="очищать папку вывода перед загрузкой (по умолчанию включено; отключить: --no-clean)")
    return p.parse_args()


def main():
    args = parse_args()
    fulltext_formats = parse_formats(args.fulltext_formats, FULLTEXT_FORMATS, "--fulltext-formats") \
        if args.full_text else []
    use_pmc = bool(args.only_full_text and fulltext_formats)
    formats_arg = args.formats if args.formats is not None else ("none" if use_pmc else "xml")
    formats = parse_formats(formats_arg, ABSTRACT_FORMATS, "--formats")

    downloader = ArticleDownloader(args.output_dir, args.email, api_key=args.api_key, clean=args.clean)
    downloader.run(
        queries=args.query or DEFAULT_QUERIES,
        max_results=args.max_results,
        limit=args.limit or None,
        formats=formats,
        fulltext_formats=fulltext_formats,
        year_from=args.year_from,
        year_to=args.year_to,
        only_full_text=args.only_full_text,
        sort=args.sort,
    )
    print("\n🎉 Готово!")


if __name__ == "__main__":
    main()