#!/usr/bin/env python3
"""
Stream-parse an enwiktionary pages-articles XML dump and write a TSV lookup:
  key = lemma_UPOS  (e.g., "approach_VERB")
  output columns:
    lemma, pos, key, ety_class, donor_langs, etym_text

Design goals:
- Never load the whole XML into memory (uses iterparse + element clearing)
- "Best effort" etymology extraction (Wiktionary is not fully regular)
- TSV output so you can build a dict later or join in pandas

Key improvements vs the earliest version:
- Handles POS headings at level 3 *and* level 4 (e.g., ====Preposition==== under ===Etymology 1===)
- Emits AUX entries when a Verb section appears to be an auxiliary/modal verb
- Normalises Latin variants (la-med, la-new, la-vul, etc.) to 'la'
- Adds OTHER class for non-(Germanic/Romance) donor languages (e.g. Greek)
- NEW: Treat ang/enm as Germanic *only when* there are no Romance/Other signals
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Set, Tuple

POS_MAP: Dict[str, str] = {
    "Noun": "NOUN",
    "Proper noun": "PROPN",
    "Verb": "VERB",
    "Adjective": "ADJ",
    "Adverb": "ADV",
    "Pronoun": "PRON",
    "Determiner": "DET",
    "Preposition": "ADP",
    "Postposition": "ADP",
    "Conjunction": "CCONJ",
    "Subordinating conjunction": "SCONJ",
    "Interjection": "INTJ",
    "Particle": "PART",
    "Numeral": "NUM",
    "Auxiliary verb": "AUX",
    "Modal verb": "AUX",
    "Article": "DET",
}

RE_H34 = re.compile(r"^(={3,4})\s*(.+?)\s*\1\s*$")

# Normalised Romance codes
ROMANCE_CODES = {
    "la", "fr", "fro", "frm", "anglo-french", "anglo-norman", "xno", "nrf",
    "pro", "it", "es", "pt", "ro",
}

# Germanic donor codes (excluding English stages, handled separately)
GERMANIC_DONOR_CODES = {
    "gem-pro", "non", "de", "nl", "nds", "gmw", "goh",
    "is", "no", "nn", "nb", "da", "sv",
    # If your dumps show these:
    "gmw-pro", "itc-pro",
}

# English historical stages: use as *fallback Germanic signals* only when no Romance/Other
ENGLISH_STAGE_CODES = {"en", "enm", "ang"}

OTHER_DONOR_PREFIXES = {"grc", "ar", "he", "fa", "ru", "tr", "zh", "ja", "ko"}
OTHER_DONOR_CODES = {"grc", "grc-koi"}

RE_DONOR_1 = re.compile(r"\{\{\s*(?:bor|der|inh|cal|lbor|ubor|learned borrowing|lb)\s*\|\s*en\s*\|\s*([a-z-]+)", re.I)
RE_DONOR_2 = re.compile(r"\{\{\s*(?:etyl|ety)\s*\|\s*([a-z-]+)\s*\|\s*en\b", re.I)
RE_DONOR_3 = re.compile(r"\{\{\s*(?:derived|der)\s*\|\s*en\s*\|\s*([a-z-]+)", re.I)

GERMANIC_KEYWORDS = [
    "Proto-Germanic", "Old Norse", "Icelandic", "Norwegian", "Danish", "Swedish",
    "Dutch", "German", "Low German",
    "Old English", "Middle English",
]
ROMANCE_KEYWORDS = [
    "Old French", "French", "Anglo-French", "Norman",
    "Latin", "Medieval Latin", "New Latin", "Vulgar Latin",
    "Italian", "Spanish", "Portuguese",
]
OTHER_KEYWORDS = [
    "Ancient Greek", "Greek",
]

RE_AUX_MARKER = re.compile(r"\b(auxiliary|modal)\b", re.I)
RE_HEAD_AUX = re.compile(r"\{\{\s*head\s*\|\s*en\s*\|\s*(auxiliary verb|modal verb)\b", re.I)
RE_AUX_TEMPLATE = re.compile(r"\{\{\s*auxiliary\s*\|\s*en\b", re.I)


def strip_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[1] if "}" in tag else tag


def load_keys(keys_path: Optional[str]) -> Optional[Set[str]]:
    if not keys_path:
        return None
    keys: Set[str] = set()
    with open(keys_path, "r", encoding="utf-8") as f:
        for line in f:
            k = line.strip()
            if k and not k.startswith("#"):
                keys.add(k)
    return keys


def normalise_lang_code(code: str) -> str:
    c = code.strip().lower()
    # Normalise Latin variants
    if c.startswith("la-") or c == "la":
        return "la"
    return c


def classify_etymology(etym_text: str) -> Tuple[str, List[str]]:
    """
    Return (class_label, donor_langs).
    class_label in {GERMANIC, ROMANCE, LATIN-FRENCH, MIXED, OTHER, UNKNOWN}
    """
    et = etym_text or ""
    donors_raw: List[str] = []

    for m in RE_DONOR_1.finditer(et):
        donors_raw.append(m.group(1))
    for m in RE_DONOR_2.finditer(et):
        donors_raw.append(m.group(1))
    for m in RE_DONOR_3.finditer(et):
        donors_raw.append(m.group(1))

    # Normalise + dedup preserve order
    seen = set()
    donors: List[str] = []
    for d in donors_raw:
        nd = normalise_lang_code(d)
        if nd and nd not in seen:
            seen.add(nd)
            donors.append(nd)

    # Keyword pseudo-donors
    if any(k in et for k in GERMANIC_KEYWORDS):
        donors.append("germanic_kw")
    if any(k in et for k in ROMANCE_KEYWORDS):
        donors.append("romance_kw")
    if any(k in et for k in OTHER_KEYWORDS):
        donors.append("other_kw")

    has_romance = any(d in ROMANCE_CODES for d in donors) or ("romance_kw" in donors)

    has_other = (
        any(d in OTHER_DONOR_CODES for d in donors) or
        any((d.split("-", 1)[0] in OTHER_DONOR_PREFIXES) for d in donors
            if d not in {"romance_kw", "germanic_kw", "other_kw"}) or
        ("other_kw" in donors)
    )

    has_germanic_strong = (
        any(d in GERMANIC_DONOR_CODES for d in donors) or
        ("germanic_kw" in donors and not has_romance)
    )

    # NEW: English-stage-only fallback → Germanic, but only if no Romance/Other signals
    has_english_stage = any(d in {"ang", "enm"} for d in donors)
    has_germanic_fallback = has_english_stage and (not has_romance) and (not has_other)

    has_latin = ("la" in donors) or ("latin" in et.lower())
    has_french = any(d in {"fr", "fro", "frm", "anglo-french", "anglo-norman", "xno", "nrf"} for d in donors) or ("french" in et.lower())

    # Decide class
    if has_romance:
        # Romance dominates unless there is strong non-English-stage Germanic donor evidence
        if has_germanic_strong:
            cls = "MIXED"
        elif has_latin and has_french:
            cls = "LATIN-FRENCH"
        else:
            cls = "ROMANCE"
    else:
        if has_germanic_strong or has_germanic_fallback:
            cls = "GERMANIC"
        elif has_other:
            cls = "OTHER"
        else:
            cls = "UNKNOWN"

    return cls, donors


def extract_english_block(wikitext: str) -> Optional[str]:
    lines = wikitext.splitlines()
    in_en = False
    out_lines: List[str] = []
    for line in lines:
        if not in_en:
            if line.strip() == "==English==":
                in_en = True
            continue
        else:
            if re.match(r"^==[^=].*==\s*$", line):
                break
            out_lines.append(line)
    if not in_en:
        return None
    return "\n".join(out_lines)


def split_sections_level34(english_block: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    lines = english_block.splitlines()
    preface_lines: List[str] = []
    sections: List[Tuple[int, str, List[str]]] = []

    cur_level: Optional[int] = None
    cur_heading: Optional[str] = None
    cur_content: List[str] = []

    started = False

    def flush_section():
        nonlocal cur_level, cur_heading, cur_content
        if cur_heading is not None and cur_level is not None:
            sections.append((cur_level, cur_heading, cur_content))
        cur_level = None
        cur_heading = None
        cur_content = []

    for line in lines:
        m = RE_H34.match(line)
        if m:
            lvl = len(m.group(1))
            heading = m.group(2).strip()
            if not started:
                started = True
            else:
                flush_section()
            cur_level = lvl
            cur_heading = heading
            cur_content = []
        else:
            if not started:
                preface_lines.append(line)
            else:
                cur_content.append(line)

    if started:
        flush_section()

    preface = "\n".join(preface_lines).strip()
    sections_out: List[Tuple[int, str, str]] = [(lvl, h, "\n".join(c).strip()) for (lvl, h, c) in sections]
    return preface, sections_out


def is_aux_verb_section(section_text: str) -> bool:
    if not section_text:
        return False
    if RE_HEAD_AUX.search(section_text):
        return True
    if RE_AUX_TEMPLATE.search(section_text):
        return True
    if RE_AUX_MARKER.search(section_text):
        return True
    return False


def parse_english_pos_etym(english_block: str) -> Dict[str, str]:
    pos_to_etym: Dict[str, str] = {}

    preface, sections = split_sections_level34(english_block)

    current_etym = ""
    preface_looks_etym = bool(
        re.search(r"\{\{\s*(?:root|inh|bor|der|etymon|etyl|ety)\b", preface) or
        any(k in preface for k in ROMANCE_KEYWORDS + GERMANIC_KEYWORDS + OTHER_KEYWORDS)
    )
    saw_etym_section = False

    for lvl, heading, content in sections:
        hlow = heading.lower()

        if hlow.startswith("etymology"):
            saw_etym_section = True
            current_etym = content
            continue

        if heading in POS_MAP:
            pos_tag = POS_MAP[heading]
            pos_to_etym[pos_tag] = current_etym

            if pos_tag == "VERB" and is_aux_verb_section(content):
                pos_to_etym["AUX"] = current_etym

            continue

    if not saw_etym_section and preface_looks_etym:
        for pos in list(pos_to_etym.keys()):
            if not pos_to_etym[pos]:
                pos_to_etym[pos] = preface

    return pos_to_etym


def tsv_escape(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to enwiktionary pages-articles XML (decompressed).")
    ap.add_argument("--out", required=True, help="Path to output TSV.")
    ap.add_argument("--keys", default=None, help="Optional file with lemma_UPOS keys to restrict output.")
    ap.add_argument("--progress-every", type=int, default=50000, help="Print progress after this many pages.")
    ap.add_argument("--max-etym-chars", type=int, default=4000, help="Trim stored etymology text to this length.")
    args = ap.parse_args()

    keys = load_keys(args.keys)

    t0 = time.time()
    page_count = 0
    written = 0

    with open(args.out, "w", encoding="utf-8") as out:
        out.write("lemma\tpos\tkey\tety_class\tdonor_langs\tetym_text\n")

        for event, elem in ET.iterparse(args.xml, events=("end",)):
            if strip_tag(elem.tag) != "page":
                continue

            page_count += 1

            try:
                ns_el = elem.find("./{*}ns")
                if ns_el is None or (ns_el.text or "") != "0":
                    continue

                title_el = elem.find("./{*}title")
                if title_el is None:
                    continue
                title = title_el.text or ""

                if ":" in title:
                    continue

                text_el = elem.find("./{*}revision/{*}text")
                wikitext = (text_el.text or "") if text_el is not None else ""

                if not wikitext or "#REDIRECT" in wikitext.upper():
                    continue

                english = extract_english_block(wikitext)
                if not english:
                    continue

                pos_to_etym = parse_english_pos_etym(english)
                if not pos_to_etym:
                    continue

                lemma = title

                for pos, etym_text in pos_to_etym.items():
                    key = f"{lemma}_{pos}"
                    if keys is not None and key not in keys:
                        continue

                    cls, donors = classify_etymology(etym_text or "")
                    donor_str = ",".join(donors)

                    et = (etym_text or "").strip()
                    if len(et) > args.max_etym_chars:
                        et = et[: args.max_etym_chars] + " …"

                    out.write(
                        f"{tsv_escape(lemma)}\t{pos}\t{tsv_escape(key)}\t{cls}\t"
                        f"{tsv_escape(donor_str)}\t{tsv_escape(et)}\n"
                    )
                    written += 1

                if page_count % args.progress_every == 0:
                    dt = time.time() - t0
                    rate = page_count / dt if dt > 0 else 0.0
                    print(
                        f"[progress] pages={page_count:,} written={written:,} rate={rate:,.1f} pages/s",
                        file=sys.stderr,
                    )

            finally:
                elem.clear()

    dt = time.time() - t0
    print(f"[done] pages={page_count:,} written={written:,} elapsed={dt:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
