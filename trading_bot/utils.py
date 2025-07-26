import re


def normalize_symbol(sym: str) -> str:
    """Return symbol in BASE_QUOTE format with underscore."""
    if not sym:
        return ""
    s = sym.replace('/', '').replace('_', '').replace(':USDT', '').upper()
    if s.endswith('USDT'):
        return f"{s[:-4]}_USDT"
    return s
