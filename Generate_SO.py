import os
import sys
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.multiprocessing as multiprocessing
from collections import OrderdDict

import easydict
from transformers import BertTokenizer, BertModel

# TBD 
from dataloader import Dataset
from utils import Config, Logger
from basemodel import BaseModel

class SE_model(BaseModel):
  def __init__(self, dataset, config, device):
      self.dataset = dataset
      self.model_conf = config
      self.device = device

  def load_so(self, mode='spade'): # load sparse output of passages, # mode : spade, custom, etc
      if mode = 'spade':
          doc_ids = self.dataset.doc_id_spade
          alpha = self.alpha
          output_expand_path = os.path.join(self.logger.log_dir, f'sparse_output_{self.cur_iter}_{len(input_pids)}_{mode}_expand.pkl')
          output_weight_path = os.path.join(self.logger.log_dir, f'sparse_output_{self.cur_iter}_{len(input_pids)}_{mode}_weight.pkl')
          self.logger.info(f"alpha: {self.alpha} (e.g., alpha * expand + (1-alpha) * weight")
        
          with open(output_expand_path, 'rb') as f:
              output_expand = pickle.load(f)
          with open(output_weight_path, 'rb') as f:
              output_weight = pickle.load(f)

          output = output_weight * (1-alpha) + output_expand * alpha
          output = output.multiply(self.df_pruning_mask)
          output = output.tocsc()
          output.eliminate_zeros()
            
      elif mode == 'custom':
          doc_ids = self.dataset.doc_id_custom
          output_expand_path = False
          output_expand_path = False
          output_custom_path = os.path.join(self.logger.log_dir, f'sparse_output_{self.cur_iter}_{len(input_pids)}_{mode}_weight.pkl')
          self.logger.info(f"alpha: {self.alpha} (e.g., alpha * expand + (1-alpha) * weight")

          with open(output_custom_path, 'rb') as f:
              output_path = pickle.load(f)
          output = output.tocsc()
          output.eliminate_zeros()

      return output


  def generate_so(self, mode='spade'): # generate and save sparse output of passages
      with torch.no_grad():
          self.eval()
          if mode == 'spade':
            doc_ids = self.dataset.doc_id_spade

            rows_weight, cols_weight, values_weight = [], [], []
            ros_expand, cols_expand, values_expand = [], [], []
            batch_doc_cols = []

            batch_loader = DataBatcher(np.arange(len(doc_ids)), 
            
