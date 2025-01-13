"""Script for bulck calculation of Tifa scores for dirs of pickle files"""

import os
import json
import pickle
import numpy as np
from PIL import Image
from tifa_utils import TifaAutoRater


def get_all_abs_filenames(dir_name):
    return [os.path.join(dir_name, f) for f in os.listdir(dir_name)]


def main():
    root_dir = '/home/royhirsch_google_com/image_editing/files/xl'
    qa_json_path = '/home/royhirsch_google_com/image_editing/sample_qa.json'
    
    
    with open(qa_json_path, 'r') as f:
        qa = json.load(f)
    tifa = TifaAutoRater()
    
    
    sample_dirs = get_all_abs_filenames(root_dir)
    for sample_dir in sample_dirs:
        sample_id = sample_dir.split('/')[-1]
        print(f'Proccessing sample id: {sample_id}')
        for sample in qa:
            if sample['id'] == sample_id:
                break

        questions = sample['questions']
        prompt = sample['prompt']
        
        # loading all the pickle files        
        all_results = []
        pickle_paths = get_all_abs_filenames(sample_dir)
        for path in pickle_paths:
            with open(path, 'rb') as f:
                results = pickle.load(f)
                all_results += results['results']
                print(f'Loaded {path}')
        n_items = len(np.unique(np.array([item['seed'] for item in all_results])))
        print(f'Loaded {n_items} unique items')

        sample_source_tifa_scores_file = os.path.join(sample_dir, f'{sample_id}_tifa_scores.pickle')
        seed2score = {}
        for item in all_results:
            image = Image.fromarray(item['image'])
            score = tifa.get_score(image, questions)
            seed2score[item['seed']] = score

        with open(sample_source_tifa_scores_file, 'wb') as f:
            pickle.dump(
                {
                    'seed2score': seed2score,
                    'questions': questions,
                    'prompt': prompt,
                },
                f)
            print(f'Saved Tifa scores to {sample_source_tifa_scores_file}')

if __name__ == '__main__':
    main()