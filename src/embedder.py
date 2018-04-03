import paddle.v2 as paddle
from squeeze import squeezenet
import gzip
import reader
import numpy as np
import random

# Initialize PaddlePaddle.
paddle.init(use_gpu=False, trainer_count=1)
with gzip.open('/book/working/params_pass_195.tar.gz', 'r') as f:
    parameters = paddle.parameters.Parameters.from_tar(f)

    

