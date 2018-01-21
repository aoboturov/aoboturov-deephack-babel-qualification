import tensorflow as tf
import numpy as np
import sys

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.layers import common_layers
from tensor2tensor.tpu import tpu_trainer_lib
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import metrics

# Enable TF Eager execution
from tensorflow.contrib.eager.python import tfe
tfe.enable_eager_execution()

# Other setup
Modes = tf.estimator.ModeKeys

ckpt_path = sys.argv[1]
fin_name = sys.argv[2]
fout_name = sys.argv[3]

# Fetch the problem
ende_problem = problems.problem("translate_ende_wmt32k")

# Get the encoders from the problem
encoders = ende_problem.feature_encoders(ckpt_path)

# Setup helper functions for encoding and decoding
def encode(input_str, output_str=None):
    """Input str to features dict, ready for inference"""
    inputs = encoders["inputs"].encode(input_str) + [1]  # add EOS id
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}

def decode(integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]
    return encoders["inputs"].decode(np.squeeze(integers))

# Create hparams and the model
model_name = "transformer"
hparams_set = "transformer_base"

hparams = tpu_trainer_lib.create_hparams(hparams_set, data_dir=ckpt_path, problem_name="translate_ende_wmt32k")

# NOTE: Only create the model once when restoring from a checkpoint; it's a
# Layer and so subsequent instantiations will have different variable scopes
# that will not match the checkpoint.
translate_model = registry.model(model_name)(hparams, Modes.EVAL)

with open(fin_name, encoding='utf-8') as fin, open(fout_name, mode='w', encoding='utf-8') as fout, tfe.restore_variables_on_create(ckpt_path):
    for inputs in fin:
        encoded_inputs = encode(inputs.strip())
        model_output = translate_model.infer(encoded_inputs, decode_length=100)
        res = decode(model_output)
        print(res, file=fout)
        fout.flush()
