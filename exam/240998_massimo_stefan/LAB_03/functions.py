# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import math

from nltk.corpus import gutenberg
from nltk.lm import NgramCounter, StupidBackoff, Vocabulary
from nltk.lm.preprocessing import flatten, padded_everygram_pipeline
import nltk


def get_dataset_and_voc():
    macbeth_sents = [[w.lower() for w in sent] for sent in gutenberg.sents('shakespeare-macbeth.txt')]
    macbeth_words = flatten(macbeth_sents)
    lex = Vocabulary(macbeth_words, unk_cutoff=2)
    macbeth_oov_sents = [list(lex.lookup(sent)) for sent in macbeth_sents]
    return macbeth_oov_sents, lex

def evaluate_base_stupidbackoff(ngram_lengths, dataset, lex, test_sents):
    padded_ngrams_oov, flat_text_oov = padded_everygram_pipeline(ngram_lengths, dataset)
    lm_oov = StupidBackoff(alpha=0.4, order=ngram_lengths)
    lm_oov.fit(padded_ngrams_oov, flat_text_oov)
    # Extract test ngrams
    ngrams, flat_text = padded_everygram_pipeline(lm_oov.order, [lex.lookup(sent) for sent in test_sents])
    ngrams = flatten(ngrams)
    # Compute PPL
    ppl = lm_oov.perplexity([x for x in ngrams if len(x) == lm_oov.order])
    print('PPL StupidBackoff:', ppl)
    
def evaluate_my_stupidbackoff(ngram_lengths, dataset, lex, test_sents):
    padded_ngrams, flat_text = padded_everygram_pipeline(ngram_lengths, dataset)
    my_lm_oov = MyStupidBackoff(alpha=0.4, order=ngram_lengths)
    my_lm_oov.fit(padded_ngrams)
    # Extract test ngrams
    ngrams, flat_text = padded_everygram_pipeline(my_lm_oov.order, [lex.lookup(sent) for sent in test_sents])
    ngrams = flatten(ngrams)
    # Compute PPL
    ppl = my_lm_oov.perplexity([x for x in ngrams if len(x) == my_lm_oov.order])
    print('PPL MyStupidBackoff:', ppl)
    
class MyStupidBackoff:
	def __init__(self, alpha:float = 0.4, order:int = 2):
		self.alpha = alpha
		self.order = order
		self.counter = None
  
	def fit(self, padded_ngrams):
		self.counter = NgramCounter(padded_ngrams)
  
	def stupid_backoff(self, ngram: tuple):
		if len(ngram) == 1:
			return self.counter[ngram[0]] / sum(self.counter.unigrams.values())
		else:
			context = ngram[:-1]
			word = ngram[-1]
			if self.counter[context][word] > 0:
				counter_curr_ngram = self.counter[context][word]
				if len(context) > 1:
					counter_lower_ngram = self.counter[context[:-1]][context[-1]]
				else:
					counter_lower_ngram = self.counter[context[0]]
				return counter_curr_ngram / counter_lower_ngram
			else:
				return self.alpha * self.stupid_backoff(ngram[1:])
  
	def perplexity(self, ngrams):
		return math.pow(2.0, self.entropy(ngrams))

	def entropy(self, ngrams):
		return -1 * sum([self.logscore(ngram[-1], ngram[:-1]) for ngram in ngrams]) / len(ngrams)

	def logscore(self, word, context):
		ngram = tuple(context[-self.order+1:] + tuple([word]))
		return math.log(self.stupid_backoff(ngram), 2)