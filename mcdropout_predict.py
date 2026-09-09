"Multi-pass inference function"

# Takes the same image through the model N times to collect N different predictions and then takes the average.
# Each pass will have a different "dropout" configuration.
import torch
import torch.nn.functional as F

def mc_dropout_predict(model, image, n_passes=50):
    model.train()  # Keeps dropout active, even though we're not training

    image = image.unsqueeze(0)  # Adds batch dimension: (1,28,28) -> (1,1,28,28)

    predictions = []
    with torch.no_grad():  # Not training so we skip the gradient tracking
        for _ in range(n_passes):
            output = model(image)      # One noisy pass, shape (1, 10)
            prob = F.softmax(output, dim=1) # Convert to probabilities after every pass and append those
            predictions.append(prob)

    predictions = torch.stack(predictions)   # Shape: (n_passes, 1, 10)

    mean_prediction = predictions.mean(dim=0)   # Average across passes -> (1, 10)
    std_prediction = predictions.std(dim=0)     # Spread across passes -> (1, 10)
    # dim=0 makes it so it's the average across the N passes not the 10 classes.

    return mean_prediction, std_prediction
