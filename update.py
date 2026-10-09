import requests
import os
import time
import shutil
from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET


class ExpertSystemsDownloader:
    def __init__(self, output_dir, email):
        self.output_dir = output_dir
        self.email = email
        self.base_search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        self.base_fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        self.headers = {
            'User-Agent': f'PyCharmMedicalArticleDownloader/1.0 (Research Tool; Contact: {email})'
        }

        # 🔥 ОЧИСТКА ПАПКИ В НАЧАЛЕ
        if Path(self.output_dir).exists():
            shutil.rmtree(self.output_dir)
            print(f"🗑️ Старая папка удалена: {self.output_dir}")

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"✅ Новая папка создана: {self.output_dir}\n")

    def search_simple(self, query, max_results=30):
        """Поиск статей на PubMed"""
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'rettype': 'xml',
            'sort': 'date'
        }

        try:
            response = requests.get(self.base_search_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()

            root = ET.fromstring(response.content)
            count = int(root.findtext('Count', 0))
            ids = [id_elem.text for id_elem in root.findall('.//Id')]

            print(f"✓ Запрос '{query}': найдено {count}, загружено {len(ids)}")
            return ids, count

        except Exception as e:
            print(f"✗ Ошибка поиска '{query}': {e}")
            return [], 0

    def download_articles(self, ids, article_format="xml"):
        """
        Скачивание полных статей

        Форматы:
        - "xml": XML с полной информацией (рекомендуется)
        - "medline": MEDLINE format
        - "fasta": Если доступно
        """
        if not ids:
            print("Нет ID для скачивания")
            return

        # Определяем параметры формата
        format_params = {
            "xml": {"rettype": "xml", "retmode": "xml", "ext": ".xml"},
            "medline": {"rettype": "medline", "retmode": "text", "ext": ".txt"},
            "json": {"rettype": "json", "retmode": "json", "ext": ".json"},
            "fasta": {"rettype": "fasta", "retmode": "text", "ext": ".fasta"}
        }

        if article_format not in format_params:
            article_format = "xml"

        fmt = format_params[article_format]
        ext = fmt["ext"]

        print(f"\n📥 Начало загрузки {len(ids)} статей в формате {article_format}...")
        print(f"📁 Папка: {self.output_dir}\n")

        for i, pubmed_id in enumerate(ids, 1):
            try:
                params = {
                    'db': 'pubmed',
                    'id': pubmed_id,
                    'rettype': fmt["rettype"],
                    'retmode': fmt["retmode"]
                }

                response = requests.get(self.base_fetch_url, params=params, headers=self.headers, timeout=10)
                response.raise_for_status()

                # Получаем название из XML если возможно
                title = "article"
                if article_format == "xml":
                    try:
                        root = ET.fromstring(response.content)
                        title_elem = root.findtext('.//ArticleTitle')
                        if title_elem:
                            title = self._sanitize_filename(title_elem[:50])
                    except:
                        pass

                # Сохраняем файл
                filename = f"{pubmed_id}_{title}{ext}"
                filepath = os.path.join(self.output_dir, filename)

                with open(filepath, 'wb') as f:
                    f.write(response.content)

                print(f"[{i}/{len(ids)}] ✓ {filename}")
                time.sleep(0.5)  # Delay между запросами

            except Exception as e:
                print(f"[{i}/{len(ids)}] ✗ PubMed ID {pubmed_id}: {e}")
                time.sleep(1)

        print(f"\n✅ Загрузка завершена! Скачано: {len(ids)} статей")

    def _sanitize_filename(self, filename):
        """Очищает имя файла от недопустимых символов"""
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '')
        return filename[:200]


# ============================================
# 🚀 ГЛАВНЫЙ КОД
# ============================================

if __name__ == "__main__":
    output_dir = r"C:\Users\admin\PyCharmMiscProject\documents"
    email = "zidzidovich@gmail.com"

    # Инициализация (автоматически очищает папку)
    downloader = ExpertSystemsDownloader(output_dir, email=email)

    # Запросы для поиска
    queries = [
        "expert systems medicine",
        "clinical decision support systems",
        "artificial intelligence diagnosis",
        "machine learning medical",
        "intelligent diagnostic",
    ]

    # Поиск
    all_ids = []
    for query in queries:
        ids, total = downloader.search_simple(query, max_results=30)
        all_ids.extend(ids)
        time.sleep(1)

    # Убираем дубликаты
    all_ids = list(set(all_ids))
    print(f"\n📊 Всего уникальных статей: {len(all_ids)}\n")

    # ВЫБИРАЙТЕ ФОРМАТ:
    # downloader.download_articles(all_ids[:50], article_format="xml")      # XML (рекомендуется) - полная инфо
    # downloader.download_articles(all_ids[:50], article_format="medline")  # MEDLINE - текстовый формат
    # downloader.download_articles(all_ids[:50], article_format="json")     # JSON - структурированный

    downloader.download_articles(all_ids[:50], article_format="xml")

    print("\n🎉 Готово!")
