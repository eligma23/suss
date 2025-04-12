try:
    from suss.constants.languages import SUPPORTED_LANGUAGES
    from suss.constants.boilerplate import PATH_EXCLUSIONS
    from suss.constants.agent import (
        MAX_MERGE_DISTANCE,
        CODE_SEARCH_LIMIT,
        FILE_SEARCH_LIMIT,
        MAX_CONTEXT_TOKENS,
        MAX_TOOL_CALLS,
        MAX_THINKING_TOKENS,
    )
except ImportError:
    from constants.languages import SUPPORTED_LANGUAGES
    from constants.boilerplate import PATH_EXCLUSIONS
    from constants.agent import (
        MAX_MERGE_DISTANCE,
        CODE_SEARCH_LIMIT,
        FILE_SEARCH_LIMIT,
        MAX_CONTEXT_TOKENS,
        MAX_TOOL_CALLS,
        MAX_THINKING_TOKENS,
    )
