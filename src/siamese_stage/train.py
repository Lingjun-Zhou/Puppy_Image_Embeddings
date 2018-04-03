import numpy as np
from siamese import SiameseNetwork, create_pairs, compute_accuracy


# Set constants
BATCH_SIZE = 128
N_EPOCHS = 20
CLASS_DIM = 120

# Construct input data
with open('/book/working/data/inter_emb.np','rb') as f:
    X_train = np.array(np.load(f), dtype=np.float32) 
with open('/book/working/data/labels.np','rb') as f:
    y_train = np.array(np.load(f), dtype=np.int8)
    digit_indices = [np.where(y_train == i)[0] for i in range(CLASS_DIM)]
tr_pairs, tr_y = create_pairs(X_train, digit_indices, CLASS_DIM)

# Construct Siamese network
model, base_network = SiameseNetwork()
model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
              batch_size=BATCH_SIZE,
              epochs=N_EPOCHS)

# Compute final accuracy on training set
y_pred = model.predict([tr_pairs[:, 0], tr_pairs[:, 1]])
tr_acc = compute_accuracy(tr_y, y_pred)
print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))

# Save
model.save('/book/working/models/siamese.h5')

