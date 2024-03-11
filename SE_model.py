import os
import pickle
import torch
import pandas an pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from time import time
from IPython import embed
from collections import defaultdict
from scipy.sparse import csc_matrix
from transformers import AutoTokenizer, AutoConfig

# Base Model import
# utils import

class SE_model():
  def __init__(self, dataset, config, device):
      self.dataset = dataset
      self.model_conf = config
      self.device = device

  def get_sparse_output():
