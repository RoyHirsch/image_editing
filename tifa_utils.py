import sys
sys.path.append('/home/royhirsch_google_com/tifa')

import os
import pickle
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from tifascore import VQAModel


class TifaAutoRater():
    def __init__(self, model_name="mplug-large"):
        self.model = VQAModel(model_name)
        self.model_name = model_name
        
    def get_scores(self, image_path, questions, debug=False):
        bin_answers = []
        for question in tqdm(questions):
            ans = self.model.multiple_choice_vqa(image_path, question['question'], choices=question['choices'])
            bin_answer = ans['multiple_choice_answer'] == question['answer']
            bin_answers.append(bin_answer)
            if debug:
                print('Q: {}\nA: {}\nPred: {}'.format(question['question'], question['answer'], ans['multiple_choice_answer']))
        return bin_answers
    
    def get_score(self, image_path, questions):
        return np.mean(self.get_scores(image_path, questions))
