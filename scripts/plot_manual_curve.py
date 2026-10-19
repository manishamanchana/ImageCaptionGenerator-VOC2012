import matplotlib.pyplot as plt

# ----- Your extracted losses -----
train_losses = [1.5947, 0.8142, 0.6650, 0.5625, 0.5015]
val_losses   = [0.9336, 0.7829, 0.7091, 0.6865]   # only 4 values available

epochs_train = range(1, len(train_losses) + 1)
epochs_val   = range(1, len(val_losses) + 1)

# ----- Plot -----
plt.figure(figsize=(10, 6))

plt.plot(epochs_train, train_losses, marker='o', linewidth=2, label="Train Loss")
plt.plot(epochs_val, val_losses, marker='o', linewidth=2, label="Validation Loss")

plt.xlabel("Epoch", fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.title("Training Curve: Train vs Validation Loss", fontsize=16)

plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

plt.savefig("figures/training_curve.png", dpi=300, bbox_inches='tight')
plt.show()

print("Saved as figures/training_curve.png")
