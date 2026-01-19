import json
import os

train_dir = "data/train"
classes = sorted(os.listdir(train_dir))

os.makedirs("models", exist_ok=True)  # Make sure models folder exists

with open("models/classes.json", "w") as f:
    json.dump(classes, f, indent=4)  # indent for readability

print("classes.json created successfully!")
