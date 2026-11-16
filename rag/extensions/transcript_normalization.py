"""Narrow, model-independent formatting normalization; no lexical corrections."""
import re
import unicodedata

POLICY_VERSION = "acronym-punctuation-v1"
POLICY = [
    "Unicode NFKC; map U+2018/U+2019/U+02BC apostrophes to ASCII apostrophe.",
    "Join bounded dotted ASCII initialisms of >=2 letters, e.g. U.S. -> US.",
    "Join bounded whitespace-separated uppercase ASCII initialisms of >=2 letters before casefold, e.g. I S A -> ISA.",
    "Unicode casefold; retain accents and symbols; do not expand words, contractions, acronyms or numbers.",
    "Replace Unicode punctuation with spaces, including hyphens; retain internal alphabetic apostrophes, numeric dot/comma separators, leading numeric minus signs and numeric percent suffixes.",
    "Collapse all whitespace and strip; retain bracketed content; never use query IDs, qrels or model identity.",
]

DOTTED = re.compile(r"(?<!\w)(?:[A-Za-z]\.\s*)+[A-Za-z]\.?(?!\w)")
SPACED = re.compile(r"(?<!\w)[A-Z](?:\s+[A-Z])+(?!\w)")
APOSTROPHES = str.maketrans({"\u2018": "'", "\u2019": "'", "\u02bc": "'"})


def normalize_transcript(text):
    text = unicodedata.normalize("NFKC", text).translate(APOSTROPHES)
    text = DOTTED.sub(lambda match: re.sub(r"[.\s]", "", match.group()), text)
    text = SPACED.sub(lambda match: "".join(match.group().split()), text).casefold()
    result = []
    for index, char in enumerate(text):
        before = text[index - 1] if index else ""
        after = text[index + 1] if index + 1 < len(text) else ""
        internal_apostrophe = char == "'" and before.isalpha() and after.isalpha()
        numeric_separator = char in ".," and before.isdecimal() and after.isdecimal()
        numeric_sign = char == "-" and after.isdecimal() and not before.isalnum()
        numeric_percent = char == "%" and before.isdecimal()
        keep = internal_apostrophe or numeric_separator or numeric_sign or numeric_percent
        result.append(" " if unicodedata.category(char).startswith("P") and not keep else char)
    return " ".join("".join(result).split())
