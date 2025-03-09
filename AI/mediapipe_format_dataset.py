import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
from mediapipe_handler import MediaPipeHandler

# TODO clean up and change to consts and dictionary
def add_new_label(example):

    workout = example['label']
    muscle_group = None

    if workout == 0: # barbell bicep curl
        muscle_group = 3 # bicep
    
    elif workout == 1: # bench press
        muscle_group = 2 # chest
        
    elif workout == 2: # chest fly machine
        muscle_group = 2 # chest

    elif workout == 3: # deadlift
        muscle_group = 7 # back

    elif workout == 4: # decline bench press
        muscle_group = 3 # biceps

    elif workout == 5: # hammer curl
        muscle_group = 3 # bicep

    elif workout == 6: # hip thrust
        muscle_group = 6 # legs

    elif workout == 7: # incline bench press
        muscle_group = 2 # chest

    elif workout == 8: # lat pulldown
        muscle_group = 7 # back

    elif workout == 9: # lateral raises
        muscle_group = 1 # shoulders
    
    elif workout == 10: # leg extensions
        muscle_group = 6 # legs

    elif workout == 11: # leg raises
        muscle_group = 4 # core

    elif workout == 12: # plank
        muscle_group = 4 # core

    elif workout == 13: # pull up
        muscle_group = 7 # back

    elif workout == 14: # push ups
        muscle_group = 2 # chest

    elif workout == 15: # romanian deadlift
        muscle_group = 6 # legs
    
    elif workout == 16: # russian twist
        muscle_group = 4 # core

    elif workout == 17: # shoulder press
        muscle_group = 1 # shoulder

    elif workout == 18: # squat
        muscle_group = 6 # glutes

    elif workout == 19: # t bar row
        muscle_group = 7 # back

    elif workout == 20: # tricep dips
        muscle_group = 5 # tricep

    elif workout == 21: # tricep pushdown
        muscle_group = 5 #tricep

    example['muscle group'] = muscle_group

    return example


ds = load_dataset("averrous/workout")
new_ds = ds.map(add_new_label)

train = new_ds['train']
test = new_ds['test']
validation = new_ds['validation']

train = pd.DataFrame(train)
test = pd.DataFrame(test)
validation = pd.DataFrame(validation)

print(type(train['image'][0]))

media_pipe_handler = MediaPipeHandler()

train_new = media_pipe_handler.pandas_add_detections_from_image(train)
test_new = media_pipe_handler.pandas_add_detections_from_image(test)
validation_new = media_pipe_handler.pandas_add_detections_from_image(validation)

print(train_new)
print(test_new)
print(validation_new)

train_new.to_csv('/Users/yasinetawfeek/Developer/DESD_AI_PATHWAY/AI/train_new.csv')
test_new.to_csv('/Users/yasinetawfeek/Developer/DESD_AI_PATHWAY/AI/test_new.csv')
validation_new.to_csv('/Users/yasinetawfeek/Developer/DESD_AI_PATHWAY/AI/validation_new.csv')