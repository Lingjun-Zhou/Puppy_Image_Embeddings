import gzip
import numpy as np
import paddle.v2 as paddle
from squeeze import squeezenet
from scipy.spatial.distance import cosine


# Set constants
dim = 128
DATA_DIM = 3 * dim * dim
CLASS_DIM = 120
BATCH_SIZE = 256

# PaddlePaddle init
paddle.init(use_gpu=True, trainer_count=1)

# Define input layers
image = paddle.layer.data(
    name="image", type=paddle.data_type.dense_vector(DATA_DIM))

# Load files
with gzip.open('/book/working/models/params_pass_47.tar.gz', 'r') as f:
    parameters = paddle.parameters.Parameters.from_tar(f)

# Configure new intermediate neural network.
new_out, intermediate = squeezenet(image, CLASS_DIM, True, True)

# Get all files
all_file_list = [line.strip().split("\t")[0] for line in open("/book/working/data/train.list")]
test_data = [(paddle.image.load_and_transform(image_file, 128 + 64, 128, False)
      .flatten().astype('float32'), )
     for image_file in all_file_list]

# Generate all embeddings
embs = []
for i in range(int(len(test_data) / 10)):
    embs += list(paddle.infer(
    output_layer=intermediate,
    parameters=parameters,
    input=test_data[i:i+10]))
    
# Save files
with open('/book/working/data/inter_emb.np','wb') as f:
    np.save(f, np.array(embs))
all_scores = [int(line.strip().split("\t")[1]) for line in open("/book/working/data/train.list")]
with open('/book/working/data/labels.np','wb') as f:
    np.save(f, np.array(all_scores))
