import logging
from typing import List, Dict
import os

logger = logging.getLogger(__name__)

from phi.tools.duckduckgo import DuckDuckGo

class EnhancedDuckDuckGo(DuckDuckGo):
    """
    Enhanced DuckDuckGo search tool that extends the base DuckDuckGo from phi.tools.duckduckgo
    """
    def duckduckgo_search(self, query: str) -> List[Dict[str, str]]:
        max_results = int(os.getenv('MAX_SEARCH_RESULTS', '5'))
        logger.info(f"Performing DuckDuckGo search for query: '{query}' with max_results: {max_results}")
        try:
            results = super().duckduckgo_search(query, max_results)
            logger.debug(f"Search completed. Found {len(results)} results.")
            return results
        except Exception as e:
            logger.error(f"Error during DuckDuckGo search: {e}", exc_info=True)
            raise
