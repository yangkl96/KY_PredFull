#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse
import utils

import tensorflow.keras as k
from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Layer, InputSpec
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Add, Flatten, Activation, BatchNormalization, LayerNormalization
# from tensorflow.keras import Model, Input

from coord_tf import CoordinateChannel2D, CoordinateChannel1D

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str,
                    help='input file path', default='example.tsv')
parser.add_argument('--batch_size', type=str,
                    help='batch size per loop', default=256)
parser.add_argument('--output', type=str,
                    help='output file path', default='example_prediction.mgf')
parser.add_argument('--model', type=str,
                    help='model file path', default='pm.h5')

args = parser.parse_args()

K.clear_session()

pm = k.models.load_model(args.model, compile=0)
pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss='cosine')

# read inputs
inputs = []
for item in pd.read_csv(args.input, sep='\t').itertuples():
    if item.Charge < 1 or item.Charge > utils.max_charge:
        print("input", item.Peptide, 'exceed max charge of', utils.max_charge, ", ignored")
        continue

    pep, mod, nterm_mod = utils.getmod(item.Peptide)

    if nterm_mod != 0:
        print("input", item.Peptide, 'has N-term modification, ignored')
        continue

    if np.any(mod != 0) and set(mod) != {0, 1}:
        print("Only Oxidation modification is supported, ignored", item.Peptide)
        continue

    inputs.append({'pep': pep, 'mod': mod, 'charge': item.Charge, 'title': item.Peptide,
                   'nce': item.NCE, 'type': utils.types[item.Type.lower()],
                   'mass': utils.fastmass(pep, 'M', item.Charge, mod=mod)})

    utils.xshape[0] = max(utils.xshape[0], len(pep) + 2)  # update xshape to match max input peptide

batch_per_loop = 64
loop_size = args.batch_size * batch_per_loop

f = open(args.output, 'w+')

while len(inputs) > 0:
    if len(inputs) >= loop_size:
        sliced_spectra = inputs[:loop_size]
        inputs = inputs[loop_size:]
    else:
        sliced_spectra = inputs
        inputs = []

    y = pm.predict(utils.input_generator(sliced_spectra, utils.preprocessor, batch_size=args.batch_size), verbose=1)
    y = np.square(y)

    f.writelines("%s\n\n" % utils.tomgf(sp, yi) for sp, yi in zip(sliced_spectra, y))

f.close()
print("Prediction finished")
