import re
import string
from pathlib import Path


class Normalizer:
	"""Normalize raw text for downstream processing."""

	def normalize(self, text):
		"""Apply full normalization in a fixed, consistent order."""
		result = text
		result = self.lowercase(result)
		result = self.remove_punctuation(result)
		result = self.remove_numbers(result)
		result = self.remove_whitespace(result)
		return result

	def load(self, folder_path):
		folder = Path(folder_path)
		texts: list[str] = []
		for txt_file in sorted(folder.glob("*.txt")):
			texts.append(txt_file.read_text(encoding="utf-8"))

		return texts

	def strip_gutenberg(self, text):
		"""Remove Gutenberg header/footer markers and surrounding text."""
		start_match = re.search(
			r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
			text,
			flags=re.IGNORECASE,
		)
		end_match = re.search(
			r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*",
			text,
			flags=re.IGNORECASE,
		)

		if start_match and end_match and start_match.end() <= end_match.start():
			return text[start_match.end() : end_match.start()].strip()
		if start_match:
			return text[start_match.end() :].strip()
		if end_match:
			return text[: end_match.start()].strip()
		return text

	def lowercase(self, text):
		"""Lowercase all text."""
		return text.lower()

	def remove_punctuation(self, text):
		"""Remove all punctuation from text."""
		return re.sub(r"[{}]".format(re.escape(string.punctuation)), "", text)

	def remove_numbers(self, text):
		"""Remove all numeric digits from text."""
		return re.sub(r"\d+", "", text)

	def remove_whitespace(self, text):
		"""Remove all whitespace characters from text."""
		return re.sub(r"\s+", "", text)

	def sentence_tokenize(self, text):
		"""Split text into a list of sentences."""
		parts = re.split(r"(?<=[.!?])\s+", text)
		return [part for part in parts if part]

	def word_tokenize(self, sentence):
		"""Split a single sentence into a list of tokens."""
		parts = re.split(r"\s+", sentence.strip())
		return [part for part in parts if part]

	def save(self, sentences, filepath):
		"""Write tokenized sentences to an output file, one sentence per line."""
		with open(filepath, "w", encoding="utf-8") as output_file:
			for sentence in sentences:
				if isinstance(sentence, (list, tuple)):
					line = " ".join(sentence)
				else:
					line = str(sentence)
				output_file.write(line + "\n")
