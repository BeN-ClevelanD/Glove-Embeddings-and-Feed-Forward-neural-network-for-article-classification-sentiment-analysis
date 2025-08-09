import re
import unicodedata
import random

STOPWORDS = set("""
the to a of in and on for that with as is at by it from an his has was will said reuters
""".split())

def remove_accents(content):
    return ''.join(
        letter for letter in unicodedata.normalize('NFKD', content)
        if not unicodedata.combining(letter)
    )
def strip_non_basic_ascii(text):
    return ''.join(c for c in text if c in 'abcdefghijklmnopqrstuvwxyz ')

def downsample_stopwords(text, keep_prob=0.2):
    tokens = text.split()
    filtered = [word for word in tokens if word not in STOPWORDS or random.random() < keep_prob]
    return ' '.join(filtered)

def clean_line(raw_line):
    result = raw_line.lower()
    result = re.sub(r'([a-z])([A-Z])', r'\1 \2', result)
    result = re.sub(r'(\d+)', r' \1 ', result)
    result = re.sub(r"'", '', result)  
    result = re.sub(r'[-]', ' ', result)
    result = re.sub(r'<[^>]+>', ' ', result)
    result = re.sub(r'&[a-zA-Z0-9#]+;', ' ', result)
    result = re.sub(r'\b(quot|nbsp|39|amp|lt|gt)\b', ' ', result)
    result = re.sub(r'\b\w*(http|www|href|html|com|net|asp|php)\w*\b', '', result)
    result = re.sub(r'\b(reuters|usatodaycom|forbescom|afp|ap|cnn|techweb|maccentral|spacecom)\b', '', result)
    result = re.sub(r'\d+', '', result)
    result = re.sub(r'[^\w\s]', ' ', result)
    result = re.sub(r'\s+', ' ', result)
    result = remove_accents(result)
    result = strip_non_basic_ascii(result)
    result = downsample_stopwords(result, keep_prob=0.2)
    tokens = result.split()
    tokens = [tok for tok in tokens if len(tok) > 1 or tok in ('a', 'i')]
    return ' '.join(tokens)


