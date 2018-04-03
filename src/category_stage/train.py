import paddle.v2 as paddle
from squeeze import squeezenet
import gzip
import reader
import numpy as np
import random

paddle.init(use_gpu=True, trainer_count=1)

DATA_DIM = 3 * 128 * 128
CLASS_DIM = 120
BATCH_SIZE = 256

# Define input layers
image = paddle.layer.data(
    name="image", type=paddle.data_type.dense_vector(DATA_DIM))
lbl = paddle.layer.data(
    name="label", type=paddle.data_type.integer_value(CLASS_DIM))

# Obtain network's softmx layer
out = squeezenet(image, CLASS_DIM, True)

# Classification cost
cost = paddle.layer.classification_cost(input=out, label=lbl)

# Create parameters
# parameters = paddle.parameters.create(cost)
with gzip.open('/book/working/models/params_pass_47.tar.gz', 'r') as f:
    parameters = paddle.parameters.Parameters.from_tar(f)

# Read training data
train_reader = paddle.batch(
    paddle.reader.shuffle(
        reader.train_reader('/book/working/data/train.list', buffered_size=1024),
        buf_size=20000),
    batch_size=BATCH_SIZE)
# Read testing data
test_reader = paddle.batch(
    reader.test_reader('/book/working/data/val.list', buffered_size=1024),
    batch_size=BATCH_SIZE)

# End batch and end pass event handler
def event_handler(event):
    # Report result of batch.
    if isinstance(event, paddle.event.EndIteration):
        print "\nPass %d, Batch %d, Cost %f, %s" % (
            event.pass_id, event.batch_id, event.cost, event.metrics)
    # Report result of pass.
    if isinstance(event, paddle.event.EndPass):
        if (event.pass_id + 1) % 4 == 0:
            # Save parameters
            with gzip.open('/book/working/models/params_pass_%d.tar.gz' % event.pass_id, 'w') as f:
                parameters.to_tar(f)
            # Report validation accuracy
            result = trainer.test(reader=test_reader)
            print "\nTest with Pass %d, %s" % (event.pass_id, result.metrics)

# Create optimizer
optimizer = paddle.optimizer.Adam(
    regularization=paddle.optimizer.L2Regularization(rate=0.0005 *
                                                       BATCH_SIZE))

# Create trainer
trainer = paddle.trainer.SGD(
    cost=cost,
    parameters=parameters,
    update_equation=optimizer)

# Train model
trainer.train(
    reader=train_reader, num_passes=200, event_handler=event_handler)
