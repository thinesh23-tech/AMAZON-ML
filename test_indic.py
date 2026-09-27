# Simple rule-based Brahmic / Indic script to Latin phonetic mapping
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Devanagari Unicode: 0x0900 - 0x097F
DEVA_MAP = {
    0x0905: 'a', 0x0906: 'aa', 0x0907: 'i', 0x0908: 'ee', 0x0909: 'u', 0x090a: 'oo',
    0x090e: 'e', 0x090f: 'e', 0x0910: 'ai', 0x0912: 'o', 0x0913: 'o', 0x0914: 'au',
    0x0915: 'k', 0x0916: 'kh', 0x0917: 'g', 0x0918: 'gh', 0x0919: 'n',
    0x091a: 'ch', 0x091b: 'chh', 0x091c: 'j', 0x091d: 'jh', 0x091e: 'n',
    0x091f: 't', 0x0920: 'th', 0x0921: 'd', 0x0922: 'dh', 0x0923: 'n',
    0x0924: 't', 0x0925: 'th', 0x0926: 'd', 0x0927: 'dh', 0x0928: 'n',
    0x092a: 'p', 0x092b: 'ph', 0x092c: 'b', 0x092d: 'bh', 0x092e: 'm',
    0x092f: 'y', 0x0930: 'r', 0x0932: 'l', 0x0933: 'l', 0x0935: 'v',
    0x0936: 'sh', 0x0937: 'sh', 0x0938: 's', 0x0939: 'h',
    0x093e: 'a', 0x093f: 'i', 0x0940: 'ee', 0x0941: 'u', 0x0942: 'oo',
    0x0947: 'e', 0x0948: 'ai', 0x094b: 'o', 0x094c: 'au',
    0x094d: '', # virama
    0x0902: 'n', 0x0901: 'n', 0x0903: 'h',
    0x0958: 'q', 0x0959: 'kh', 0x095a: 'gh', 0x095b: 'z', 0x095c: 'd', 0x095d: 'dh', 0x095e: 'f', 0x095f: 'y',
}

# Tamil Unicode: 0x0B80 - 0x0BFF
TAMIL_MAP = {
    0x0B85: 'a', 0x0B86: 'aa', 0x0B87: 'i', 0x0B88: 'ee', 0x0B89: 'u', 0x0B8A: 'oo',
    0x0B8E: 'e', 0x0B8F: 'e', 0x0B90: 'ai', 0x0B92: 'o', 0x0B93: 'o', 0x0B94: 'au',
    0x0B95: 'k', 0x0B99: 'n', 0x0B9A: 'ch', 0x0B9C: 'j', 0x0B9E: 'n',
    0x0B9F: 't', 0x0BA3: 'n', 0x0BA4: 't', 0x0BA8: 'n', 0x0BA9: 'n',
    0x0BAA: 'p', 0x0BAE: 'm', 0x0BAF: 'y', 0x0BB0: 'r', 0x0BB1: 'r',
    0x0BB2: 'l', 0x0BB3: 'l', 0x0BB4: 'zh', 0x0BB5: 'v', 0x0BB7: 'sh',
    0x0BB8: 's', 0x0BB9: 'h',
    0x0BBE: 'a', 0x0BBF: 'i', 0x0BC0: 'ee', 0x0BC1: 'u', 0x0BC2: 'oo',
    0x0BC6: 'e', 0x0BC7: 'e', 0x0BC8: 'ai', 0x0BCA: 'o', 0x0BCB: 'o', 0x0BCC: 'au',
    0x0BCD: '', # pulli / virama
}

def transliterate_indic(text):
    out = []
    for ch in text:
        cp = ord(ch)
        if cp in DEVA_MAP:
            out.append(DEVA_MAP[cp])
        elif cp in TAMIL_MAP:
            out.append(TAMIL_MAP[cp])
        elif 0x0C00 <= cp <= 0x0C7F: # Telugu offset ~ Devanagari - 0x0900 + 0x0C00
            rel = cp - 0x0C00 + 0x0900
            out.append(DEVA_MAP.get(rel, ''))
        elif 0x0C80 <= cp <= 0x0CFF: # Kannada offset
            rel = cp - 0x0C80 + 0x0900
            out.append(DEVA_MAP.get(rel, ''))
        elif 0x0980 <= cp <= 0x09FF: # Bengali offset
            rel = cp - 0x0980 + 0x0900
            out.append(DEVA_MAP.get(rel, ''))
        else:
            out.append(ch)
    return "".join(out)

test_words = [
    "एसएस फूड प्राइवेट लिमिटेड",
    "रेड वेंचर्स प्राइवेट लिमिटेड",
    "होटल एंटरप्राइजेज लिमिटेड",
    "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி",
    "ಕರ್ನಾಟಕ",
    "हरियाणा"
]

for w in test_words:
    print(f"{w:30s} -> {transliterate_indic(w)}")
