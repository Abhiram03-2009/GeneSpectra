import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for server
import matplotlib.pyplot as plt
from scipy.stats import iqr
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io
import base64
