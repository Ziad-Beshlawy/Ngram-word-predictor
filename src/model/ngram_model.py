import os
from collections import Counter
import json


class NGramModel:
	"""N-gram language model."""

	def build_vocab(self, token_file):
		"""Collect unique words and replace rare words with <UNK>."""
		unk_threshold = int(os.getenv("UNK_THRESHOLD", 4))
		
		word_counts = Counter()
		with open(token_file, "r", encoding="utf-8") as f:
			for line in f:
				tokens = line.strip().split()
				word_counts.update(tokens)
		
		vocab = []
		for word, count in word_counts.items():
			if count >= unk_threshold:
				vocab.append(word)

		vocab.append("<UNK>")
		
		self.vocab = vocab
		return vocab

	def build_counts_and_probabilities(self, token_file):
		"""Build MLE probabilities for 1..N where higher orders are next-word conditionals."""
		ngram_order = int(os.getenv("NGRAM_ORDER", 4))

		tokens = []
		with open(token_file, "r", encoding="utf-8") as f:
			for line in f:
				tokens.extend(line.strip().split())

		unigram_counts = Counter(tokens)
		probabilities = {order: {} for order in range(1, ngram_order + 1)}

		unigram_total = sum(unigram_counts.values())
		for word, count in unigram_counts.items():
			if unigram_total > 0:
				probabilities[1][word] = count / unigram_total
			else:
				probabilities[1][word] = 0.0

		for order in range(2, ngram_order + 1):
			context_counts = Counter()
			next_word_counts = Counter()

			for i in range(len(tokens) - order + 1):
				ngram = tokens[i : i + order]
				context = " ".join(ngram[:-1])
				next_word = ngram[-1]
				context_counts[context] += 1
				next_word_counts[(context, next_word)] += 1

			for (context, next_word), count in next_word_counts.items():
				if context not in probabilities[order]:
					probabilities[order][context] = {}
				history_count = context_counts[context]
				if history_count > 0:
					probabilities[order][context][next_word] = count / history_count
				else:
					probabilities[order][context][next_word] = 0.0

		self.probabilities = probabilities
		return probabilities

	def load(self, model_path, vocab_path):
		"""Load model and vocab from JSON files into the instance."""
		with open(model_path, "r", encoding="utf-8") as f:
			raw = json.load(f)
		self.probabilities = {int(order): probs for order, probs in raw.items()}

		with open(vocab_path, "r", encoding="utf-8") as f:
			self.vocab = json.load(f)

	def save_vocab(self, vocab_path):
		"""Save vocabulary list to a JSON file."""
		os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
		with open(vocab_path, "w", encoding="utf-8") as f:
			json.dump(self.vocab, f)

	def save_model(self, model_path):
		"""Save all probability tables to a JSON file."""
		os.makedirs(os.path.dirname(model_path), exist_ok=True)
		saveable = {str(order): probs for order, probs in self.probabilities.items()}
		with open(model_path, "w", encoding="utf-8") as f:
			json.dump(saveable, f)

	def lookup(self, context):
		"""Backoff lookup from highest-order context down to 1-gram."""
		words = context.strip().split()

		for order in range(len(words), 0, -1):
			key = " ".join(words[-order:])
			if key in self.probabilities.get(order + 1, {}):
				return self.probabilities[order + 1][key]

		return self.probabilities.get(1, {})
