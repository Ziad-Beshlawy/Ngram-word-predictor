from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel


def test_predictor_init_stores_preloaded_instances():
	model = NGramModel()
	normalizer = Normalizer()

	predictor = Predictor(model, normalizer)

	assert predictor.model is model
	assert predictor.normalizer is normalizer


def test_map_oov_replaces_unknown_words_with_unk():
	model = NGramModel()
	model.vocab = ["the", "cat", "<UNK>"]
	predictor = Predictor(model, Normalizer())

	result = predictor.map_oov("the dog cat")

	assert result == "the <UNK> cat"


def test_map_oov_keeps_known_words():
	model = NGramModel()
	model.vocab = ["hello", "world", "<UNK>"]
	predictor = Predictor(model, Normalizer())

	result = predictor.map_oov("hello world")

	assert result == "hello world"


def test_map_oov_returns_empty_string_for_empty_context():
	model = NGramModel()
	model.vocab = ["<UNK>"]
	predictor = Predictor(model, Normalizer())

	result = predictor.map_oov("")

	assert result == ""


def test_normalize_calls_normalizer_and_returns_last_context_window(monkeypatch):
	class StubNormalizer:
		def __init__(self):
			self.called_with = []

		def normalize(self, text):
			self.called_with.append(text)
			return text.lower()

	monkeypatch.setenv("NGRAM_ORDER", "4")
	model = NGramModel()
	normalizer = StubNormalizer()
	predictor = Predictor(model, normalizer)

	result = predictor.normalize("Raw Input")

	assert normalizer.called_with == ["Raw", "Input"]
	assert result == "raw input"


def test_predict_next_chains_normalize_map_oov_and_lookup(monkeypatch):
	class StubNormalizer:
		def normalize(self, text):
			return "context word one"

	class StubModel:
		def __init__(self):
			self.vocab = ["context", "word", "one", "<UNK>"]
			self.lookup_called_with = None

		def lookup(self, context):
			self.lookup_called_with = context
			return {"next": 0.5, "words": 0.3, "here": 0.2}

	monkeypatch.setenv("NGRAM_ORDER", "4")
	model = StubModel()
	predictor = Predictor(model, StubNormalizer())

	result = predictor.predict_next("input text", k=2)

	assert model.lookup_called_with == "context word one"
	assert result == [("next", 0.5), ("words", 0.3)]


def test_predict_next_returns_empty_for_k_zero(monkeypatch):
	class StubNormalizer:
		def normalize(self, text):
			return "one two"

	class StubModel:
		vocab = ["one", "two", "<UNK>"]

		def lookup(self, context):
			return {"word": 0.5}

	monkeypatch.setenv("NGRAM_ORDER", "3")
	predictor = Predictor(StubModel(), StubNormalizer())

	result = predictor.predict_next("input", k=0)

	assert result == []


def test_predict_next_returns_all_when_k_exceeds_vocabulary(monkeypatch):
	class StubNormalizer:
		def normalize(self, text):
			return "a b"

	class StubModel:
		vocab = ["a", "b", "<UNK>"]

		def lookup(self, context):
			return {"x": 0.6, "y": 0.4}

	monkeypatch.setenv("NGRAM_ORDER", "3")
	predictor = Predictor(StubModel(), StubNormalizer())

	result = predictor.predict_next("input", k=100)

	assert len(result) == 2
	assert result == [("x", 0.6), ("y", 0.4)]


def test_predict_next_unknown_only_context_returns_unk(monkeypatch):
	class StubNormalizer:
		def normalize(self, text):
			return text.lower()

	class StubModel:
		vocab = ["known", "<UNK>"]

		def lookup(self, context):
			return {"the": 0.4, "and": 0.3, "of": 0.2}

	monkeypatch.setenv("NGRAM_ORDER", "3")
	predictor = Predictor(StubModel(), StubNormalizer())

	result = predictor.predict_next("zzzz qqqq", k=3)

	assert result == [("<UNK>", 1.0)]
