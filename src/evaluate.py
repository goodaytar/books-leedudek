from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch

def evaluate(model, data_loader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in data_loader:

            inputs = {k: v.to(device) for k, v in batch.items() if k not in ["idx", "labels"]}
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits
            
            probs = torch.sigmoid(logits.squeeze())
            # preds = torch.argmax(probs, dim=1).cpu().numpy()
            preds = torch.round(probs)
            
            # Store predictions and true labels
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predictions, average='binary')
    conf_matrix = confusion_matrix(true_labels, predictions)

    return accuracy, precision, recall, f1, conf_matrix
