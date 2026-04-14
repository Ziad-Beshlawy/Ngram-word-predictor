"""Command-line interface for the n-gram predictor."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor
from src.model.ngram_model import NGramModel


def run_dataprep():
	"""Normalize raw training texts and save tokenized output."""
	print("Running dataprep...")
	
	train_raw_dir = os.getenv("TRAIN_RAW_DIR", "data/raw/train/")
	train_tokens_path = os.getenv("TRAIN_TOKENS", "data/processed/train_tokens.txt")
	
	normalizer = Normalizer()
	
	# Load raw texts from directory
	raw_texts = normalizer.load(train_raw_dir)
	print(f"Loaded {len(raw_texts)} raw text files.")
	
	# Tokenize, normalize, and collect sentences
	all_sentences = []
	for raw_text in raw_texts:
		# Strip Gutenberg markers
		cleaned = normalizer.strip_gutenberg(raw_text)
		
		# Sentence tokenize first (before normalizing to preserve structure)
		sentences = normalizer.sentence_tokenize(cleaned)
		
		# Word tokenize and normalize each sentence
		for sentence in sentences:
			tokens = normalizer.word_tokenize(sentence)
			# Normalize each token individually
			normalized_tokens = [normalizer.normalize(token) for token in tokens]
			# Filter out empty tokens
			normalized_tokens = [t for t in normalized_tokens if t]
			if normalized_tokens:
				all_sentences.append(normalized_tokens)
	
	# Save tokenized sentences
	Path(train_tokens_path).parent.mkdir(parents=True, exist_ok=True)
	normalizer.save(all_sentences, train_tokens_path)
	print(f"Saved {len(all_sentences)} tokenized sentences to {train_tokens_path}")


def run_model():
	"""Build n-gram model and save model and vocabulary."""
	print("Running model training...")
	
	train_tokens_path = os.getenv("TRAIN_TOKENS", "data/processed/train_tokens.txt")
	model_path = os.getenv("MODEL", "data/model/model.json")
	vocab_path = os.getenv("VOCAB", "data/model/vocab.json")
	
	model = NGramModel()
	
	# Build vocabulary
	vocab = model.build_vocab(train_tokens_path)
	print(f"Built vocabulary with {len(vocab)} words.")
	
	# Build probabilities
	probs = model.build_counts_and_probabilities(train_tokens_path)
	print(f"Built {len(probs)} n-gram probability tables.")
	
	# Save model and vocabulary
	Path(model_path).parent.mkdir(parents=True, exist_ok=True)
	model.save_vocab(vocab_path)
	model.save_model(model_path)
	print(f"Saved model to {model_path}")
	print(f"Saved vocabulary to {vocab_path}")


def run_inference():
	"""Interactive CLI for next-word prediction."""
	print("Loading model...")
	
	model_path = os.getenv("MODEL", "data/model/model.json")
	vocab_path = os.getenv("VOCAB", "data/model/vocab.json")
	top_k = int(os.getenv("TOP_K", 3))
	
	# Instantiate and load model
	model = NGramModel()
	model.load(model_path, vocab_path)
	print(f"Loaded model with {len(model.vocab)} vocabulary words.")
	
	# Instantiate normalizer and predictor
	normalizer = Normalizer()
	predictor = Predictor(model, normalizer)
	
	print("Starting inference loop. Type 'quit' to exit.\n")
	
	while True:
		try:
			user_input = input("> ").strip()
			
			if not user_input:
				continue
			
			if user_input.lower() == "quit":
				print("Goodbye.")
				break
			
			# Get predictions
			predictions = predictor.predict_next(user_input, k=top_k)
			
			# Extract just the words and format for display
			predicted_words = [word for word, prob in predictions]
			print(f"Predictions: {predicted_words}\n")
			
		except KeyboardInterrupt:
			print("\n\nGoodbye.")
			break
		except Exception as e:
			print(f"Error: {e}\n")


def main():
	"""Parse arguments and run the requested pipeline step."""
	parser = argparse.ArgumentParser(
		description="N-gram predictor CLI with dataprep, model training, and inference modes."
	)
	parser.add_argument(
		"--step",
		choices=["dataprep", "model", "inference", "all"],
		default="inference",
		help="Which step to run: dataprep, model, inference, or all (default: inference)",
	)
	
	args = parser.parse_args()
	
	# Load environment variables
	load_dotenv("config/.env")
	
	# Run requested step(s)
	if args.step == "dataprep":
		run_dataprep()
	elif args.step == "model":
		run_model()
	elif args.step == "inference":
		run_inference()
	elif args.step == "all":
		run_dataprep()
		run_model()
		run_inference()


if __name__ == "__main__":
	main()
