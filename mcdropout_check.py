import torch
from data import test_data
from mcdropout_model import MCDropoutCNN
from mcdropout_predict import mc_dropout_predict

model = MCDropoutCNN()
model.load_state_dict(torch.load('mcdropout_weights.pth'))
model.eval()

n_images_to_check = 5

for i in range(n_images_to_check):
    image, label = test_data[i]

    # --- Deterministic single prediction (dropout off) ---
    with torch.no_grad():
        single_output = model(image.unsqueeze(0))
        single_pred = single_output.argmax(dim=1).item()

    # --- MC-Dropout multi-pass prediction (dropout on, averaged) ---
    mean_pred, std_pred = mc_dropout_predict(model, image, n_passes=50)
    mc_pred = mean_pred.argmax(dim=1).item()
    mc_confidence = mean_pred[0, mc_pred].item()
    mc_uncertainty = std_pred[0, mc_pred].item()

    print(f"Image {i} | True label: {label}")
    print(f"  Single-pass prediction:     {single_pred}")
    print(f"  MC-Dropout prediction:      {mc_pred}  (confidence: {mc_confidence:.3f}, uncertainty: {mc_uncertainty:.3f})")
    print()