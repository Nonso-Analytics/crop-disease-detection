import json
import os

train_dir = "data/train"
classes = sorted(os.listdir(train_dir))

# Wrap in a dict
class_dict = {"classes": classes}

with open("models/classes.json", "w") as f:
    json.dump(class_dict, f)
