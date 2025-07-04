## Importing all dependencies
from transformers import pipeline, AutoTokenizer
from  datasets import load_dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from tensorflow.keras.optimizers import Adam
import evaluate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import Trainer

# loading the imdb dataset 
try:
    dataset = load_dataset("imdb")
except Exception as e:
    print(f"Error loading dataset: {e}")
    raise

# defining the base model because after each iteration of training we wil be using the trained model to fine tune further
base_model_name = "google-bert/bert-base-uncased"

# Creating instance of the tokenizer for the particular model that we will be fine tuning
try:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name, num_labels=2)
except Exception as e:
    print(f"Error loading model/tokenizer: {e}")
    raise

# function to tokenize the input text
def tokenize(examples):
    if "text" not in examples:
      raise ValueError("Expected 'text' key in dataset examples.")
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# function to evaluate model after each round f training
accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    metrics = {}
    try:
      metrics.update(accuracy_metric.compute(predictions=predictions, references=labels))
      metrics.update(precision_metric.compute(predictions=predictions, references=labels, average='weighted'))
      metrics.update(recall_metric.compute(predictions=predictions, references=labels, average='weighted'))
      metrics.update(f1_metric.compute(predictions=predictions, references=labels, average='weighted'))
    except Exception as e:
       print(f'Error calculating evaluation metrics: {e}')
       raise
    return metrics


# loop for fine tuning model
all_metrics = []

for i in range(2):
  print(f"\n Starting run {i + 1}/2...")
  metric = evaluate.load("accuracy")

  small_train = dataset["train"].shuffle(seed=np.random.randint(0, 10000)).select(range(10000)).map(tokenize, batched=True)
  small_eval = dataset["test"].shuffle(seed=np.random.randint(0, 10000)).select(range(10000)).map(tokenize, batched=True)

  training_args = TrainingArguments(
    output_dir="./trainer_outputs",
    eval_strategy="epoch",
    learning_rate = 2e-5
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train,
    eval_dataset=small_eval,
    compute_metrics=compute_metrics,
    )
  try:
    trainer.train()
    eval_metrics = trainer.evaluate()
    print(f" Run {i + 1} evaluation metrics:", eval_metrics)
    all_metrics.append(eval_metrics)
  except Exception as e:
     print(f"Error during training: {e}")
     raise
  

# saving the fine tuned parameters
model.save_pretrained("./fine_tuned_model")

# plotting the evaluation metrics
df = pd.DataFrame(all_metrics)
df["run"] = df.index + 1

plt.figure(figsize=(12, 6))
plt.plot(df["run"], df["eval_accuracy"], marker='o', label="Accuracy")
plt.plot(df["run"], df["eval_loss"], marker='x', label="Loss")
plt.xlabel("Run")
plt.ylabel("Metric Value")
plt.title("Evaluation Metrics Across Runs")
plt.legend()
plt.show()
df.to_csv("metrics.csv", index=False)


# testing the model on a text example
text = " I liked the movie but it was really stereotypical so I would not recommend it "
model = AutoModelForSequenceClassification.from_pretrained("/content/fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)

print(predictions)



























