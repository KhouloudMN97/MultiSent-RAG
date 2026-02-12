import wikipedia
from typing import List, Dict, Optional


class WikipediaDocument:
    """
    Container for a Wikipedia document used in the retrieval knowledge base.
    """

    def __init__(self, text: str, metadata: Dict):
        self.text = text
        self.metadata = metadata


class WikipediaLoader:
    """
    Multilingual Wikipedia extractor for sentiment-related topics.

    - Retrieves up to 100 documents per language.
    - Truncates each document to ~1000 characters (as stated in the paper).
    - Used as unstructured knowledge source for MultiSent-RAG.
    """

    def __init__(self, language: str):
        self.language = language
        wikipedia.set_lang(language)

    def search_titles(self, query: str, max_results: int = 100) -> List[str]:
        """
        Search Wikipedia for relevant page titles.
        """
        return wikipedia.search(query, results=max_results)

    def fetch_page(self, title: str) -> Optional[WikipediaDocument]:
        """
        Fetch a Wikipedia page and return a truncated document.
        """
        try:
            page = wikipedia.page(title)

            metadata = {
                "source": "Wikipedia",
                "title": page.title,
                "language": self.language,
                "description": f"Summary of {page.title} from Wikipedia."
            }

            # Truncate to approximately 1000 characters (paper requirement)
            truncated_content = page.content[:1000]

            return WikipediaDocument(
                text=truncated_content,
                metadata=metadata
            )

        except wikipedia.exceptions.DisambiguationError:
            return None
        except wikipedia.exceptions.PageError:
            return None
        except Exception:
            return None

    def extract_topics(
        self,
        topics: List[str],
        max_results: int = 100
    ) -> List[WikipediaDocument]:
        """
        Extract documents for a list of sentiment-related topics.
        """
        documents = []

        for topic in topics:
            titles = self.search_titles(topic, max_results=max_results)

            for title in titles:
                doc = self.fetch_page(title)
                if doc is not None:
                    documents.append(doc)

        return documents
