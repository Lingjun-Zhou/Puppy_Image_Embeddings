from siamese import SiameseNetwork

# Construct Siamese network
model, base_network = SiameseNetwork()
model.load_weights('/book/working/models/siamese.h5')

def intermediate(embs):
    return base_network.predict(embs)

