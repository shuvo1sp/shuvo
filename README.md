# shuvo

# Step 1: Install necessary libraries
!pip install transformers datasets torch matplotlib scikit-learn seaborn

# Step 2: Import required libraries
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding, pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

# Step 3: Load online dataset (Hugging Face emotion dataset)
dataset = load_dataset("[]n")

# Step 4: Load tokenizer and model (Pre-trained DistilBERT for emotion classification)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Step 5: Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Step 6: Prepare dataset for PyTorch
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")

# Step 7: Split dataset into train & test
train_dataset = tokenized_datasets["train"]
test_dataset = tokenized_datasets["test"]

# Step 8: Load pre-trained model for classification
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=6)

# Step 9: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
)

# Step 10: Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# Step 11: Train the model
trainer.train()

# Step 12: Evaluate the model
eval_results = trainer.evaluate()
print(f"Evaluation Results: {eval_results}")

# Step 13: Predict the class for new text samples
emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

new_texts = [
    "I am feeling very happy today!",
    "This is so frustrating and makes me angry!",
    "I am sad and feeling down.",
    "I am so excited for my vacation!",
    "I feel neutral about this event."
]

predictions = [emotion_classifier(text)[0] for text in new_texts]

# Step 14: Print Predictions
for text, pred in zip(new_texts, predictions):
    print(f"Text: {text}\nPredicted Emotion: {pred['label']} with Confidence: {pred['score']:.4f}\n")

# Step 15: Compute Accuracy and Confusion Matrix
y_true = np.array(test_dataset["labels"].tolist())
y_pred = []

for text in dataset["test"]["text"]:
    pred = emotion_classifier(text)[0]['label']
    y_pred.append(int(pred.split("_")[-1]))  # Convert label to numerical format

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

# Step 16: Visualize Model Performance
plt.figure(figsize=(12, 4))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(trainer.state.log_history, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

# Confusion Matrix Plot
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["sadness", "joy", "love", "anger", "fear", "surprise"], yticklabels=["sadness", "joy", "love", "anger", "fear", "surprise"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
# Step 16: Visualize Model Performance
plt.figure(figsize=(12, 4))

# Loss Plot
plt.subplot(1, 2, 1)
# Extract loss values from the log history
loss_values = [log_entry['loss'] for log_entry in trainer.state.log_history if 'loss' in log_entry]
plt.plot(loss_values, label="Loss") # Plot loss values against epoch number (implicitly assumed)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()

# Confusion Matrix Plot
plt.subplot(1, 2, 2)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["sadness", "joy", "love", "anger", "fear", "surprise"], yticklabels=["sadness", "joy", "love", "anger", "fear", "surprise"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
Step 13 (Modified): Enter your own text and get predictions

emotion_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define emotion types and emojis
emotion_types = {
    "sadness": "Sadness 😢",
    "joy": "Joy😊",
    "love": "Love❤️",
    "anger": "Anger😡",
    "fear": "Fear😨",
    "surprise": "Surprise😮"
}

while True:
    user_text = input("Enter text to analyze (or type 'exit' to quit): ")
    if user_text.lower() == 'exit':
        break

    prediction = emotion_classifier(user_text)[0]
    emotion_label = prediction['label'].split('_')[-1]  # Extract emotion label (e.g., 'joy', 'anger')

    # Convert numerical label to emotion label
    emotion_mapping = {
        '0': 'sadness',
        '1': 'joy',
        '2': 'love',
        '3': 'anger',
        '4': 'fear',
        '5': 'surprise'
    }

    emotion_label = emotion_mapping.get(emotion_label, emotion_label) # Get emotion label from mapping

    # Get emoji for the predicted emotion
    emotion_emoji = emotion_types.get(emotion_label, "Neutral 😐")  # Default to neutral if not found

    print(f"Predicted Emotion: {emotion_emoji} with Confidence: {prediction['score']:.4f}\n")
