# Trah Classification
This project utilize Vision Transformer to clasiy trash image dataset. The reason why Vision Transformer is choosed over traditional CNNs because its can they use a self-attention mechanism that allows them to attend to any part of the image, regardless of its location. Its also can be trained on smaller datasets since the dataset only have 5054 samples.

# Step To Reproduce This Project's Model
- install dependencies

  ```
  pip install -r requirements.txt
  ```
- load the dataset
  ```
  from datasets import load_dataset
  dataset = load_dataset("garythung/trashnet")
  ```
- preprocess the dataset
  ```
  dataset = dataset["train"].train_test_split(test_size=0.3, shuffle=False)

  from transformers import AutoImageProcessor
  checkpoint = "google/vit-base-patch16-224-in21k"
  image_processor = AutoImageProcessor.from_pretrained(checkpoint)

  from tensorflow import keras
  from tensorflow.keras import layers
  
  size = (image_processor.size["height"], image_processor.size["width"])
  
  train_data_augmentation = keras.Sequential(
      [
          layers.RandomCrop(size[0], size[1]),
          layers.Rescaling(scale=1.0 / 127.5, offset=-1),
          layers.RandomFlip("horizontal"),
          layers.RandomRotation(factor=0.02),
          layers.RandomZoom(height_factor=0.2, width_factor=0.2),
      ],
      name="train_data_augmentation",
  )
  
  val_data_augmentation = keras.Sequential(
      [
          layers.CenterCrop(size[0], size[1]),
          layers.Rescaling(scale=1.0 / 127.5, offset=-1),
      ],
      name="val_data_augmentation",
  )

  import numpy as np
  import tensorflow as tf
  from PIL import Image
  
  
  def convert_to_tf_tensor(image: Image):
      np_image = np.array(image)
      tf_image = tf.convert_to_tensor(np_image)
      # `expand_dims()` is used to add a batch dimension since
      # the TF augmentation layers operates on batched inputs.
      return tf.expand_dims(tf_image, 0)
  
  
  def preprocess_train(example_batch):
      """Apply train_transforms across a batch."""
      images = [
          train_data_augmentation(convert_to_tf_tensor(image.convert("RGB").resize((224,224)))) for image in example_batch["image"]
      ]
      example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
      return example_batch
  
  
  def preprocess_val(example_batch):
      """Apply val_transforms across a batch."""
      images = [
          val_data_augmentation(convert_to_tf_tensor(image.convert("RGB").resize((224,224)))) for image in example_batch["image"]
      ]
      example_batch["pixel_values"] = [tf.transpose(tf.squeeze(image)) for image in images]
      return example_batch  
  ```
  
- appy the preprocess method
  ```
  dataset["train"].set_transform(preprocess_train)
  dataset["test"].set_transform(preprocess_val)
  
  from transformers import DefaultDataCollator
  data_collator = DefaultDataCollator(return_tensors="tf")

  # converting our train dataset to tf.data.Dataset
  tf_train_dataset = dataset["train"].to_tf_dataset(
      columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
  )
  
  # converting our test dataset to tf.data.Dataset
  tf_eval_dataset = dataset["test"].to_tf_dataset(
      columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
  )
  ```
  
- intialize some metrics evaluation
  ```
  import evaluate
  accuracy = evaluate.load("accuracy")

  import numpy as np
  def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)

  from transformers import create_optimizer
  batch_size = 32
  num_epochs = 10
  num_train_steps = len(dataset["train"]) * num_epochs
  learning_rate = 3e-5
  weight_decay_rate = 0.01
  
  optimizer, lr_schedule = create_optimizer(
      init_lr=learning_rate,
      num_train_steps=num_train_steps,
      weight_decay_rate=weight_decay_rate,
      num_warmup_steps=0,
  )
  ```
  
- load the model
  ```
  from transformers import TFAutoModelForImageClassification

  model = TFAutoModelForImageClassification.from_pretrained(
    "suramadu08/vit-trash",
    id2label=id2label,
    label2id=label2id,
  )
  ```
  
- compile the model
  ```
  from tensorflow.keras.losses import SparseCategoricalCrossentropy
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  model.compile(optimizer=optimizer, loss=loss)
  ```
  
- set the callbacks
  ```
  from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

  metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)
  callbacks = [metric_callback]
  ```
  
- Finally train the model
  ```
  model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=num_epochs, callbacks=callbacks)
  ```
