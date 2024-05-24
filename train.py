import huggingface_hub, wandb
from datasets import load_dataset
import tensorflow as tf
import random
from transformers import AutoImageProcessor
from tensorflow import keras
from tensorflow.keras import layers
from transformers import DefaultDataCollator
import evaluate
import numpy as np
import os
from PIL import Image
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
from transformers import create_optimizer
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from transformers.keras_callbacks import KerasMetricCallback, PushToHubCallback

HF_KEY = os.environ["HF_KEY"]
WANDB_API_KEY = os.environ["WANDB_API_KEY"]

def main():
    huggingface_hub.login(token=HF_KEY)
    wandb.login(key=WANDB_API_KEY)

    # load dataset
    dataset = load_dataset("garythung/trashnet")

    dataset = dataset["train"].train_test_split(test_size=0.3, shuffle=False)

    labels = dataset["train"].features["label"].names
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    

    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

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


    def convert_to_tf_tensor(image: Image):
        np_image = np.array(image)
        tf_image = tf.convert_to_tensor(np_image)
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

    dataset["train"].set_transform(preprocess_train)
    dataset["test"].set_transform(preprocess_val)

    data_collator = DefaultDataCollator(return_tensors="tf")

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    

    batch_size = 32
    num_epochs = 3
    num_train_steps = len(dataset["train"]) * num_epochs
    learning_rate = 3e-5
    weight_decay_rate = 0.01

    optimizer, lr_schedule = create_optimizer(
        init_lr=learning_rate,
        num_train_steps=num_train_steps,
        weight_decay_rate=weight_decay_rate,
        num_warmup_steps=0,
    )

    from transformers import TFAutoModelForImageClassification

    model = TFAutoModelForImageClassification.from_pretrained(
    "suramadu08/vit-trash",
    id2label=id2label,
    label2id=label2id,
    )

    tf_train_dataset = dataset["train"].to_tf_dataset(
        columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )

    # converting our test dataset to tf.data.Dataset
    tf_eval_dataset = dataset["test"].to_tf_dataset(
        columns="pixel_values", label_cols="label", shuffle=True, batch_size=batch_size, collate_fn=data_collator
    )
    

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)


    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_eval_dataset)

    wandb.init(
            project="vit-trash-classification",
            config={
                "epoch": 3,
            },
    )
    config = wandb.config

    wandb_callbacks = [
            WandbMetricsLogger(),
            WandbModelCheckpoint(filepath="my_model_{epoch:02d}"),
            metric_callback
    ]

    model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=3, callbacks=wandb_callbacks)

    model.evaluate(tf_eval_dataset)

    model.summary()

    model.push_to_hub("vision-trash")

    wandb.finish()
    
if __name__ == "__main__":

    main()
