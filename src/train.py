
import os
import mlflow
import mlflow.pytorch
from src import evaluate

from transformers import AdamW
from tqdm import tqdm
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

# Set up MLflow experiment
mlflow.set_tracking_uri('http://localhost:5003')
mlflow.set_experiment('Austen classifier')


# Create a function to save the confusion matrix as a text file
def save_confusion_matrix(conf_matrix, filename="confusion_matrix.txt"):
    np.savetxt(filename, conf_matrix, fmt='%d')


# Training loop with corrected artifact logging
def train(model, model_name, train_loader, eval_loader, test_loader, device, num_epochs=1):

    step = 0

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=5e-5)
    model.to(device)

    with mlflow.start_run():
        for epoch in range(num_epochs):
            running_loss = 0.0
            for batch in tqdm(train_loader):
                model.train()
                # Move batch to device
                inputs = {k: v.to(device) for k, v in batch.items() if k not in ["idx", "labels"]}
                labels = batch['labels'].to(device)

                # Forward pass
                outputs = model(**inputs, labels=labels)
                loss= loss_fn(outputs.logits.squeeze(-1), labels.float())
                # loss = outputs.loss.float()

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                running_loss += loss_item
   
                mlflow.log_metric("step_train_loss", loss_item, step=step)
                # Evaluate and log evaluation performance
                # return model, eval_loader, device, evaluate.evaluate
                eval_accuracy, eval_precision, eval_recall, eval_f1, eval_conf_matrix = evaluate.evaluate(model, eval_loader, device)
                mlflow.log_metric("step_eval_accuracy", eval_accuracy, step=step)
                mlflow.log_metric("step_eval_precision", eval_precision, step=step)
                mlflow.log_metric("step_eval_recall", eval_recall, step=step)
                mlflow.log_metric("step_eval_f1", eval_f1, step=step)
                step+=1

            # Calculate average loss
            avg_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

            # Log the loss to MLflow
            mlflow.log_metric("epoch_train_loss", avg_loss, step=epoch)


            # Evaluate and log evaluation performance
            eval_accuracy, eval_precision, eval_recall, eval_f1, eval_conf_matrix = evaluate.evaluate(model, eval_loader, device)
            print(f"Epoch {epoch + 1}/{num_epochs}, Eval Accuracy: {eval_accuracy}, Precision: {eval_precision}, Recall: {eval_recall}, F1: {eval_f1}")
            mlflow.log_metric("epoch_eval_accuracy", eval_accuracy, step=epoch)
            mlflow.log_metric("epoch_eval_precision", eval_precision, step=epoch)
            mlflow.log_metric("epoch_eval_recall", eval_recall, step=epoch)
            mlflow.log_metric("epoch_eval_f1", eval_f1, step=epoch)

            # Save and log the confusion matrix for evaluation set
            save_confusion_matrix(eval_conf_matrix, "eval_confusion_matrix.txt")
            mlflow.log_artifact("eval_confusion_matrix.txt")

        # Evaluate on test set and log test performance
        test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix = evaluate.evaluate(model, test_loader, device)
        print(f"Test Accuracy: {test_accuracy}, Precision: {test_precision}, Recall: {test_recall}, F1: {test_f1}")
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1", test_f1)
        mlflow.log_param('model_name', model_name)
        # mlflow.log()
        # Save and log the confusion matrix for test set
        save_confusion_matrix(test_conf_matrix, "test_confusion_matrix.txt")
        mlflow.log_artifact("test_confusion_matrix.txt")

        # Log the final model
        mlflow.pytorch.log_model(model, "model")
