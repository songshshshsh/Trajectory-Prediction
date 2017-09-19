import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description="Prediction")
parser.add_argument("--batch_size", type=int, default=500)

args = parser.parse_args()

class Agent(object):
	"""docstring for Agent"""
	def __init__(self, arg):
		self.generator = self.create_generator()
		self.discriminator = self.create_discriminator()

	def create_generator():
		return

	def create_discriminator():
		return