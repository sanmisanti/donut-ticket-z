import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments
# from datasets import load_metric # opcional para métricas - DEPRECATED
# Para métricas, ahora usar: import evaluate
from PIL import Image