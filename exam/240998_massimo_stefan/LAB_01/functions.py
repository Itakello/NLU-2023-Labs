# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

from collections import Counter

import nltk
import spacy


def get_corpus(text_file = 'milton-paradise.txt') -> tuple[str, list[str], list[str]]:
    nltk.download('gutenberg')
    from nltk.corpus import gutenberg
    m_chars = gutenberg.raw(text_file)
    m_words = gutenberg.words(text_file)
    m_sents = gutenberg.sents(text_file)
    return m_chars, m_words, m_sents

def print_statistics(chars, words, sents) -> None:
  word_lens = [len(word) for word in words]
  sent_lens = [len(sent) for sent in sents]
  
  print("TOTAL chars: ", len(chars))
  print("TOTAL words:", len(words))
  print("TOTAL sents: ", len(sents))
  print("AVG char per word: ", round(sum(word_lens) / len(words)))
  print("AVG word per sent: ",round(sum(sent_lens) / len(sents)))
  print("AVG word len: ", round(sum(word_lens)/len(word_lens)))
  print("MIN word len: ", min(word_lens))
  print("MAX word len: ", max(word_lens))
  print("AVG sent len: ", round(sum(sent_lens)/len(sent_lens)))
  print("MIN sent len: ", min(sent_lens))
  print("MAX sent len: ", max(sent_lens))
  
def get_stats_spacy(chars:str) -> tuple[str, list[str], list[str]]:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(chars, disable=["tagger", "ner"])
    chars_spacy = [char for char in doc.text]
    words_spacy = [token.text for token in doc]
    sentes_spacy = [sent for sent in doc.sents]
    return chars_spacy, words_spacy, sentes_spacy
    
def get_stats_nltk(chars:str) -> tuple[str, list[str], list[str]]:
    chars_nltk = chars
    words_nltk = nltk.word_tokenize(chars)
    sentes_nltk = nltk.sent_tokenize(chars)
    return chars_nltk, words_nltk, sentes_nltk
    
def compute_low_lexicons(m_words, words_spacy, words_nltk) -> None:
    lexicon_reference = set([w.lower() for w in m_words])
    lexicon_spacy = set([w.lower() for w in words_spacy])
    lexicon_nltk = set([w.lower() for w in words_nltk])
    print("--Lexicon sizes for each version--")
    print("Lexicon size reference: ",len(lexicon_reference))
    print("Lexicon size spacy: ",len(lexicon_spacy))
    print("Lexicon size nltk: ",len(lexicon_nltk))
    
def _nbest(d, n=1):
	"""
	get n max values from a dict
	:param d: input dict (values are numbers, keys are stings)
	:param n: number of values to get (int)
	:return: dict of top n key-value pairs
	"""
	return dict(sorted(d.items(), key=lambda item: item[1], reverse=True)[:n])

def compute_freq_distributions(m_words, words_spacy, words_nltk, n=10) -> None:
    m_lowercase_freq_list = Counter([w.lower() for w in m_words])
    m_lowercase_freq_list_spacy = Counter([w.lower() for w in words_spacy])
    m_lowercase_freq_list_nltk = Counter([w.lower() for w in words_nltk])

    print("--Frequency diustribution for each version--")
    print(f"Frequency distribution reference of top {n} frequencies: ", _nbest(m_lowercase_freq_list, n))
    print(f"Frequency distribution spacy of top {n} frequencies: ", _nbest(m_lowercase_freq_list_spacy, n))
    print(f"Frequency distribution nltk of top {n} frequencies: ", _nbest(m_lowercase_freq_list_nltk, n))