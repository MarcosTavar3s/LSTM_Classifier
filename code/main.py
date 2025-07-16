from lstm import classifier_model
import mlflow

model = classifier_model(dataset_dir="UCF50", classes_list=["WalkingWithDog","Skiing", "Swing", "Diving", "Mixing", "HorseRace", "HorseRiding"])
model.architecture()

mlflow.set_tracking_uri("http://localhost:5000")

# set the experiment idcl
mlflow.set_experiment(experiment_id="568692922124802970")

mlflow.autolog()

# Use the model to make predictions on the test dataset.
model.create_dataset()
model.train(epochs=30)
predictions = model.evaluate()

model.save_model()
# model.load_model("model.keras")
# model.predict(video_file_path="UCF50/Diving/v_Diving_g01_c03.avi")
