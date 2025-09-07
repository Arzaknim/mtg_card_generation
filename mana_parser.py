# MANA_MAP = {
#     "W": "white mana",
#     "U": "blue mana",
#     "B": "black mana",
#     "R": "red mana",
#     "G": "green mana",
#     "C": "colorless mana",
#     "X": "X generic mana",
#     "S": "snow mana",
#     # Hybrid mana
#     "W/U": "white or blue mana",
#     "W/B": "white or black mana",
#     "U/B": "blue or black mana",
#     "U/R": "blue or red mana",
#     "B/R": "black or red mana",
#     "B/G": "black or green mana",
#     "R/G": "red or green mana",
#     "R/W": "red or white mana",
#     "G/W": "green or white mana",
#     "G/U": "green or blue mana",
#     # Phyrexian
#     "W/P": "white phyrexian mana",
#     "U/P": "blue phyrexian mana",
#     "B/P": "black phyrexian mana",
#     "R/P": "red phyrexian mana",
#     "G/P": "green phyrexian mana",
# }
#
# import re
#
#
# def parse_mana_cost(mana_str: str) -> str:
#     # Extract symbols like {G}, {U}, {2}, etc.
#     symbols = re.findall(r"\{([^}]*)\}", mana_str)
#
#     parsed = []
#     for sym in symbols:
#         if sym.isdigit():  # numeric
#             parsed.append(f"{sym} generic mana")
#         elif sym in MANA_MAP:
#             parsed.append(MANA_MAP[sym])
#         else:
#             parsed.append(sym.lower())  # fallback
#     return ", ".join(parsed)
#
#
# def replace_mana_symbols(text: str) -> str:
#     """
#     Replaces all mana symbols like {G}, {1}, {X}, {W/U}, {B/P} etc.
#     with standardized tokens, while keeping the rest of the text intact.
#     """
#     # Match {anything}
#     return re.sub(r"\{([^}]*)\}", lambda m: f"{{{m.group(1)}}}", text)
#
#
# if __name__ == '__main__':
#     # Example usage:
#     oracle = "Add {G}{G}. Exile target creature unless its controller pays {1}."
#     print(replace_mana_symbols(oracle))


import re

# Mapping for simple mana
MANA_MAP = {
    "W": "white mana",
    "U": "blue mana",
    "B": "black mana",
    "R": "red mana",
    "G": "green mana",
    "C": "colorless mana",
    "X": "X generic mana",
    "S": "snow mana ",
    "T": "tap ",   # {T} symbol
    "Q": "untap "  # {Q} symbol
}

def mana_to_text(symbol: str) -> str:
    # Handle hybrid mana like {W/U}
    if "/" in symbol:
        parts = symbol.split("/")
        if "P" in parts:  # Phyrexian mana
            pure_mana = MANA_MAP.get(parts[0], parts[0])
            return pure_mana.replace(" ", " phyrexian ")
        else:
            return "/".join(MANA_MAP.get(p, p) for p in parts)

    # Handle numbers
    if symbol.isdigit():
        return f"{symbol} generic"

    # Handle normal single symbols
    return MANA_MAP.get(symbol, symbol)

def replace_mana_symbols(text: str) -> str:
    text = text.replace('}{', '}, {')
    return re.sub(r"\{([^}]*)\}", lambda m: mana_to_text(m.group(1)), text)


if __name__ == '__main__':
    # Example
    oracle = "{T}: Add {G}{G}. Exile target creature unless its controller pays {B/P}. {W/U} or {B/P}"
    print(replace_mana_symbols(oracle))