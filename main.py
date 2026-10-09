import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# ← Установите переменную окружения ДО инициализации Ollama классов
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

# Остальные импорты...
from typing import List, Dict
import chardet
import time
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS


class RAGSystem:
    def __init__(
            self,
            model_name: str = "gemma4",
            embedding_model: str = "nomic-embed-text",
            ollama_host: str = "http://127.0.0.1:11434"
    ):
        """
        Инициализация RAG системы

        Args:
            model_name: Название модели в Ollama для генерации
            embedding_model: Модель для создания embeddings
            ollama_host: URL Ollama сервера
        """
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.ollama_host = ollama_host

        # Обновление переменной окружения
        os.environ["OLLAMA_HOST"] = ollama_host

        # Проверка подключения к Ollama
        self._check_ollama_connection()

        # Инициализация компонентов БЕЗ base_url (используется OLLAMA_HOST)
        self.llm = OllamaLLM(
            model=model_name,
            temperature=0.7
        )
        self.embeddings = OllamaEmbeddings(
            model=embedding_model
        )
        self.vector_store = None
        self.qa_chain = None

        # Параметры для разделения текста
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _check_ollama_connection(self) -> None:
        """Проверка подключения к Ollama серверу"""
        print(f"Проверка подключения к Ollama на {self.ollama_host}...")

        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
                if response.status_code == 200:
                    print("✓ Ollama сервер доступен!")
                    models = response.json().get("models", [])
                    print(f"Доступные модели: {len(models)}")
                    return
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    print(f"✗ Не удалось подключиться. Повторная попытка через {retry_delay}с...")
                    time.sleep(retry_delay)
                else:
                    print("\n" + "=" * 60)
                    print("ОШИБКА: Ollama сервер не запущен!")
                    print("=" * 60)
                    print("\nДля запуска Ollama:")
                    print("1. Скачайте с https://ollama.ai")
                    print("2. Откройте приложение Ollama")
                    print("3. Или запустите из терминала: ollama serve")
                    print("\nДля загрузки моделей:")
                    print("  ollama pull gemma4")
                    print("  ollama pull nomic-embed-text")
                    print("=" * 60 + "\n")
                    sys.exit(1)
            except Exception as e:
                print(f"✗ Ошибка при проверке: {str(e)}")
                sys.exit(1)

    def detect_encoding(self, file_path: str) -> str:
        """Определение кодировки файла"""
        with open(file_path, 'rb') as f:
            result = chardet.detect(f.read())
        return result['encoding'] or 'utf-8'

    def load_txt_file(self, file_path: str) -> str:
        """Загрузка TXT файла с автоматическим определением кодировки"""
        encoding = self.detect_encoding(file_path)
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            for fallback_encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=fallback_encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            raise Exception(f"Не удалось определить кодировку для {file_path}")

    def load_pdf_file(self, file_path: str) -> str:
        """Загрузка PDF файла"""
        try:
            from PyPDF2 import PdfReader
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PdfReader(f)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            return text
        except ImportError:
            print("PyPDF2 не установлен. Установите: pip install PyPDF2")
            return ""

    def load_docx_file(self, file_path: str) -> str:
        """Загрузка DOCX файла"""
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except ImportError:
            print("python-docx не установлен. Установите: pip install python-docx")
            return ""

    def load_html_file(self, file_path: str) -> str:
        """Загрузка HTML файла"""
        try:
            from bs4 import BeautifulSoup
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                text = soup.get_text()
            return text
        except ImportError:
            print("beautifulsoup4 не установлен. Установите: pip install beautifulsoup4")
            return ""

    def load_xml_file(self, file_path: str) -> str:
        """
        Загрузка XML файла с парсингом структуры

        Поддерживает два режима:
        1. Встроенный ElementTree (без зависимостей)
        2. BeautifulSoup (если установлен, для более сложных XML)
        """
        try:
            # Попытка использовать BeautifulSoup для лучшей обработки
            from bs4 import BeautifulSoup
            encoding = self.detect_encoding(file_path)
            with open(file_path, 'r', encoding=encoding) as f:
                soup = BeautifulSoup(f.read(), 'xml')
                text = soup.get_text(separator="\n", strip=True)
            return text
        except ImportError:
            # Резервный вариант с встроенным ElementTree
            try:
                encoding = self.detect_encoding(file_path)
                tree = ET.parse(file_path)
                root = tree.getroot()
                text = self._extract_xml_text(root)
                return text
            except ET.ParseError as e:
                print(f"Ошибка при парсинге XML: {str(e)}")
                return ""
        except Exception as e:
            print(f"Ошибка при загрузке XML файла: {str(e)}")
            return ""

    def _extract_xml_text(self, element, level: int = 0) -> str:
        """
        Рекурсивное извлечение текста из XML элемента
        Сохраняет структуру документа
        """
        text_parts = []

        # Добавляем текст элемента
        if element.text and element.text.strip():
            indent = "  " * level
            text_parts.append(f"{indent}{element.tag}: {element.text.strip()}")

        # Рекурсивно обрабатываем дочерние элементы
        for child in element:
            text_parts.append(self._extract_xml_text(child, level + 1))

        # Добавляем tail текст (текст после закрывающего тега)
        if element.tail and element.tail.strip():
            text_parts.append(element.tail.strip())

        return "\n".join([t for t in text_parts if t])

    def load_document(self, file_path: str) -> str:
        """Загрузка документа в зависимости от формата"""
        file_ext = Path(file_path).suffix.lower()

        print(f"Загрузка: {file_path}")

        if file_ext == '.txt':
            return self.load_txt_file(file_path)
        elif file_ext == '.pdf':
            return self.load_pdf_file(file_path)
        elif file_ext == '.docx':
            return self.load_docx_file(file_path)
        elif file_ext in ['.html', '.htm']:
            return self.load_html_file(file_path)
        elif file_ext == '.xml':
            return self.load_xml_file(file_path)
        else:
            print(f"Формат {file_ext} не поддерживается. Пытаюсь загрузить как TXT...")
            return self.load_txt_file(file_path)

    def load_documents_from_directory(self, directory_path: str, supported_formats: List[str] = None) -> List[Document]:
        """
        Загрузка документов из директории
        """
        if supported_formats is None:
            supported_formats = ['.txt', '.pdf', '.docx', '.html', '.htm', '.xml']

        documents = []
        directory = Path(directory_path)

        if not directory.exists():
            print(f"Директория {directory_path} не существует!")
            return documents

        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_formats:
                try:
                    content = self.load_document(str(file_path))
                    if content.strip():
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(file_path),
                                "filename": file_path.name,
                                "file_type": file_path.suffix
                            }
                        )
                        documents.append(doc)
                        print(f"✓ Успешно загружен: {file_path.name}")
                except Exception as e:
                    print(f"✗ Ошибка при загрузке {file_path.name}: {str(e)}")

        return documents

    def create_vector_store(self, documents: List[Document]) -> None:
        """Создание векторного хранилища из документов"""
        if not documents:
            print("Нет документов для обработки!")
            return

        print(f"\nОбработка {len(documents)} документов...")

        chunks = []
        for doc in documents:
            split_chunks = self.text_splitter.split_text(doc.page_content)
            for chunk in split_chunks:
                chunks.append(Document(
                    page_content=chunk,
                    metadata=doc.metadata
                ))

        print(f"Создано {len(chunks)} чанков текста")

        print("Создание embeddings (может занять время)...")
        try:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            print("✓ Векторное хранилище создано!")
        except Exception as e:
            print(f"✗ Ошибка при создании embeddings: {str(e)}")
            raise

    def create_qa_chain(self) -> None:
        """Создание цепи для ответов на вопросы"""
        if self.vector_store is None:
            print("Сначала загрузите документы и создайте векторное хранилище!")
            return

        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})

        prompt = ChatPromptTemplate.from_template(
            """Ты помощник, который отвечает на вопросы на основе предоставленного контекста.

Контекст:
{context}

Вопрос: {question}

Ответ:"""
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
                {"context": retriever | (lambda docs: format_docs(docs)), "question": RunnablePassthrough()}
                | prompt
                | self.llm
        )

        print("✓ QA цепь создана!")

    def query(self, question: str) -> Dict:
        """Получение ответа на вопрос"""
        if self.qa_chain is None:
            print("Сначала создайте QA цепь!")
            return {}

        try:
            answer = self.qa_chain.invoke(question)

            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(question)
            sources = [doc.metadata["source"] for doc in docs]

            return {
                "answer": answer,
                "sources": sources
            }
        except Exception as e:
            print(f"✗ Ошибка при получении ответа: {str(e)}")
            return {}

    def save_vector_store(self, path: str) -> None:
        """Сохранение векторного хранилища"""
        if self.vector_store is None:
            print("Нет векторного хранилища для сохранения!")
            return

        self.vector_store.save_local(path)
        print(f"✓ Векторное хранилище сохранено в {path}")

    def load_vector_store(self, path: str) -> None:
        """Загрузка векторного хранилища"""
        try:
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✓ Векторное хранилище загружено из {path}")
        except Exception as e:
            print(f"✗ Ошибка при загрузке: {str(e)}")

    def load_vector_store(self, path: str) -> None:
        """Загрузка векторного хранилища"""
        try:
            self.vector_store = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"✓ Векторное хранилище загружено из {path}")
        except Exception as e:
            print(f"✗ Ошибка при загрузке: {str(e)}")


# Пример использования
if __name__ == "__main__":
    docs_directory = "./documents"
    vector_store_path = "./vector_store"

    # Создание системы
    rag = RAGSystem(
        model_name="gemma4",
        embedding_model="nomic-embed-text",
        ollama_host="http://127.0.0.1:11434"
    )

    documents = rag.load_documents_from_directory(docs_directory)

    if documents:
        rag.create_vector_store(documents)
        rag.save_vector_store(vector_store_path)
        rag.create_qa_chain()

        print("\n" + "=" * 50)
        print("RAG система готова. Введите 'exit' для выхода")
        print("=" * 50 + "\n")

        while True:
            question = input("Вопрос: ").strip()

            if question.lower() == 'exit':
                break

            if question:
                result = rag.query(question)
                print(f"\nОтвет: {result['answer']}")
                print(f"Источники: {result['sources']}\n")
    else:
        print("Не удалось загрузить документы!")
