from lstm import classifier_model

# Instatiate the LSTM class
model = classifier_model(dataset_dir="UCF50", classes_list=["WalkingWithDog","Skiing", "Swing", "Diving", "Mixing", "HorseRace", "HorseRiding"])
model.architecture()

# Create dataset
model.create_dataset()

# Training step
model.train(epochs=30)

# Save model and evaluate its performance during training/testing
model.save_model()
model.evaluate()
model.save_metrics()
