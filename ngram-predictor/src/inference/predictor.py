import os


class Predictor:
	"""Inference wrapper around a pre-loaded language model and normalizer."""

	def __init__(self, model, normalizer):
		"""Accept pre-loaded collaborators without performing file I/O."""
		self.model = model
		self.normalizer = normalizer

	def map_oov(self, context):
		"""Replace out-of-vocabulary words in the context with <UNK>."""
		vocab = set(self.model.vocab)
		words = context.strip().split()
		mapped_words = [word if word in vocab else "<UNK>" for word in words]
		return " ".join(mapped_words)

	def normalize(self, text):
		"""Normalize text and return the trailing context window."""
		# Normalize token-by-token so whitespace boundaries are preserved for context.
		tokens = text.strip().split()
		normalized_tokens = [self.normalizer.normalize(token) for token in tokens]
		normalized_tokens = [token for token in normalized_tokens if token]
		ngram_order = int(os.getenv("NGRAM_ORDER", 4))
		context_size = max(ngram_order - 1, 0)
		if context_size == 0:
			return ""
		return " ".join(normalized_tokens[-context_size:])

	def predict_next(self, text, k):
		"""Orchestrate normalize → map_oov → lookup → return top-k by probability."""
		context = self.normalize(text)
		mapped_context = self.map_oov(context)
		if mapped_context:
			context_words = mapped_context.split()
			if context_words and all(word == "<UNK>" for word in context_words):
				return [("<UNK>", 1.0)][:k]
		probabilities = self.model.lookup(mapped_context)
		sorted_words = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
		return sorted_words[:k]
