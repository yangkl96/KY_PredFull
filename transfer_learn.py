#!/usr/bin/env python3

import numpy as np
from pyteomics import mgf
import argparse
import os
import pandas as pd
import sys
import utils
import datetime
import re
import random

random.seed(123)

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
import keras_tuner
from coord_tf import CoordinateChannel1D


def parse_spectra(sps):
    # ratio constants for NCE
    cr = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}

    db = []

    for sp in sps:
        param = sp['params']

        c = int(str(param['charge'][0])[0])

        if 'seq' in param:
            pep = param['seq']
        else:
            pep = param['title']

        if 'pepmass' in param:
            mass = param['pepmass'][0]
        else:
            mass = float(param['parent'])

        if 'hcd' in param:
            try:
                hcd = param['hcd']
                if hcd[-1] == '%':
                    hcd = float(hcd)
                elif hcd[-2:] == 'eV':
                    hcd = float(hcd[:-2])
                    hcd = hcd * 500 * cr[c] / mass
                else:
                    raise Exception("Invalid type!")
            except:
                hcd = 0
        else:
            hcd = 0

        mz = sp['m/z array']
        it = sp['intensity array']

        db.append({'pep': pep, 'charge': c,
                   'mass': mass, 'mz': mz, 'it': it, 'nce': hcd, 'type': utils.types[args.fragmentation.lower()]})

    return db


def readmgf(fn):
    file = open(fn, "r")
    data = mgf.read(file, convert_arrays=1, read_charges=False,
                    dtype='float32', use_index=False)

    codes = parse_spectra(data)
    file.close()
    return codes


def spectrum2vector(mz_list, itensity_list, mass, bin_size, charge):
    itensity_list = itensity_list / np.max(itensity_list)

    vector = np.zeros(utils.dim, dtype='float32')

    mz_list = np.asarray(mz_list)

    indexes = mz_list / bin_size
    indexes = np.around(indexes).astype('int32')

    for i, index in enumerate(indexes):
        if index < len(vector):  # peaks at 2000 m/z ignored
            vector[index] += itensity_list[i]

    # normalize
    vector = np.sqrt(vector)

    # remove precursors, including isotropic peaks
    for delta in (0, 1, 2):
        precursor_mz = mass + delta / charge
        if precursor_mz > 0 and precursor_mz < 2000:
            vector[round(precursor_mz / bin_size)] = 0

    return vector


parser = argparse.ArgumentParser()
parser.add_argument('--filtered_mgf', type=str, help='mgf to train on', default="")
parser.add_argument('--mgf_folder', type=str, help='folder with mgf files')
parser.add_argument('--msms_folder', type=str, help='folder with msms identifications')  #
parser.add_argument('--base_model', type=str, help='base model to transfer learn on', default="pm.h5")
parser.add_argument('--fragmentation', type=str, help='fragmentation method (HCD, CID, etc)', default="hcd")  #
parser.add_argument('--min_score', type=int, help='minimum Andromeda score', default=150)
parser.add_argument('--nce', type=int, help='normalized collision energy', default=30)
parser.add_argument('--epochs', type=int, help='number of epochs to transfer learn model', default=50)
parser.add_argument('--hyperband_iterations', type=int, help='number of hyperband_iterations to transfer learn model',
                    default=2)
parser.add_argument('--batch_size', type=int, help='batch size for training', default=32)
parser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
parser.add_argument('--from_scratch', type=bool, help='whether to retrain entire model', default=False)
parser.add_argument('--out', type=str, help='filename to save the transfer learned model', default="")
parser.add_argument('--processing_only', type=bool,
                    help='whether to just do preprocessing of files', default=False)
parser.add_argument('--tuner_search', type=bool,
                    help='whether to do hyperparameter tuner searching', default=False)

args = parser.parse_args()
if not os.path.exists(args.base_model):
    print(
        "pm.h5 model is missing. Please download from https://drive.google.com/drive/folders/1Ca3HdV-w8TZPRa9KhPBbjrTtGSmtEIsn")
    sys.exit(0)
if os.path.getsize(args.base_model) < 1000:
    print(
        "You might have wrong pm.h5 model. Please download from https://drive.google.com/drive/folders/1Ca3HdV-w8TZPRa9KhPBbjrTtGSmtEIsn")
    sys.exit(0)

if ".h5" not in args.out:
    args.out = args.out + "/" + str(datetime.datetime.now()).replace(" ", "_") + ".h5"
    print("setting model output location to " + args.out)
if args.filtered_mgf == "":
    # preparing annotated mgf files first
    filtered_mgf = []
    msms_folders = args.msms_folder.split(",")
    mgf_folders = args.mgf_folder.split(",")
    #iterate through mgf/df_dict pairs
    #need to only keep best PSM across all files
    for msms_f, mgf_f in zip(msms_folders, mgf_folders):
        print(msms_f + ";" + mgf_f)
        for root, dirs, files in os.walk(msms_f):
            for file in files:
                if file == "msms.txt":
                    print(os.path.join(root, file))
                    if os.path.getsize(os.path.join(root, file)) == 0:
                        continue
                    df = pd.read_csv(os.path.join(root, file), sep="\t")
                    df = df[df["Score"] >= args.min_score]
                    df = df[df["Fragmentation"].str.lower() == args.fragmentation.lower()]
                    if df.shape[0] == 0:
                        continue

                    df_dict = {}


                    def get_dict_entry(x):
                        df_dict[x["Scan number"]] = x["Modified sequence"][
                                                    1:len(x["Modified sequence"]) - 1]  # .replace("M(ox)", "m")


                    df.apply(lambda x: get_dict_entry(x), axis=1)

                    # read in potential mgf files
                    mgf_file = mgf_f + "/" + df["Raw file"].values[0] + ".mgf"
                    if not os.path.exists(mgf_file):
                        continue
                    spectra = readmgf(mgf_file)

                    for sp in spectra:
                        # check if sp is suitable for inclusion, based on predfull data preprocessing standards, and has ID
                        length = len(sp["mz"])
                        scan_num = int(sp["pep"].split(" ")[0].split(".")[1])
                        if length > 20:  # underfragmented
                            if length < 500:  # overfragmented
                                if scan_num in df_dict.keys():
                                    # can't seem to find entries with >200 ppm difference?
                                    sp["params"] = {}
                                    sp["params"]["charge"] = sp["charge"]
                                    sp["params"]["pepmass"] = sp["mass"]
                                    sp["params"]["seq"] = df_dict[scan_num]
                                    # MAX_PEPTIDE_LENGTH = np.max(MAX_PEPTIDE_LENGTH, df_dict[scan_num] + 2)
                                    sp["params"]["nce"] = args.nce  # can try making charge dependent, and add a default
                                    sp["m/z array"] = sp["mz"]
                                    sp["intensity array"] = sp["it"]
                                    filtered_mgf.append(sp)
    #            if len(filtered_mgf) > 0:
    #                break
    print("writing filtered mgf at filtered_" + args.fragmentation + ".mgf")
    mgf.write(filtered_mgf, output="filtered_" + args.fragmentation + ".mgf", file_mode="w")
    args.filtered_mgf = "filtered_" + args.fragmentation + ".mgf"
if args.processing_only:
    print("processing done")
    sys.exit(0)
K.clear_session()

print('Reading mgf...')
old_spectra = readmgf(args.filtered_mgf)
spectra = []
for sp in old_spectra:
    add = True
    mods = [m.start() for m in re.finditer("\\(", sp["pep"])]
    for mod in mods:
        if sp["pep"][mod + 1:mod + 4] != "ox)":
            add = False
    if add:
        spectra.append(sp)
# old_spectra = None

for sp in spectra:
    utils.xshape[0] = max(utils.xshape[0], len(sp["pep"]) + 2)  # update xshape to match max input peptide

validation_split_idx = int(len(spectra) * 9 / 10)


# adapted from https://stackoverflow.com/questions/62916904/failed-copying-input-tensor-from-cpu-to-gpu-in-order-to-run-gatherve-dst-tensor
class DataGenerator(Sequence):
    def __init__(self, embedding_set, meta_set, y_set, batch_size):
        self.embeddings, self.metas, self.y = embedding_set, meta_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.embeddings) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = [self.embeddings[idx * self.batch_size:(idx + 1) * self.batch_size],
                   self.metas[idx * self.batch_size:(idx + 1) * self.batch_size]]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        return batch_x, batch_y


class DataGeneratorLite(Sequence):
    def __init__(self, embedding_set, meta_set, y_set, batch_size):
        self.embeddings, self.metas, self.y = embedding_set, meta_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.embeddings) / float(self.batch_size)))

    def __getitem__(self, idx):
        subset_embeddings = self.embeddings[idx * self.batch_size:(idx + 1) * self.batch_size]
        subset_metas = self.metas[idx * self.batch_size:(idx + 1) * self.batch_size]
        subset_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = [np.zeros((len(subset_embeddings), self.embeddings[0].shape[0], self.embeddings[0].shape[1])),
                   np.zeros((len(subset_metas), self.metas[0].shape[0], self.metas[0].shape[1]))]
        batch_y = np.zeros((len(subset_y), len(self.y[0])))

        for j in range(len(subset_embeddings)):
            batch_x[0][j] = subset_embeddings[j]
            batch_x[1][j] = subset_metas[j]
            batch_y[j] = subset_y[j]

        return batch_x, batch_y


print("Processing")
idxs = list(range(len(spectra)))
random.shuffle(idxs)
y = [spectrum2vector(spectra[i]['mz'], spectra[i]['it'], spectra[i]['mass'], utils.precision, spectra[i]['charge'])
     for i in idxs]
infos = utils.preprocessor(spectra)
embeddings = [infos[0][i] for i in idxs]
metas = [infos[1][i] for i in idxs]

#train_gen = None
#test_gen = None

try:  # when memory is not an issue
    y_array = np.zeros((len(y), len(y[0])))
    for i in range(len(y)):
        y_array[i] = y[i]
    embeddings_array = np.zeros((len(embeddings), embeddings[0].shape[0], embeddings[0].shape[1]))
    for i in range(len(embeddings)):
        embeddings_array[i] = embeddings[i]
    metas_array = np.zeros((len(metas), metas[0].shape[0], metas[0].shape[1]))
    for i in range(len(metas)):
        metas_array[i] = metas[i]

    train_gen = DataGenerator(embeddings_array[0:validation_split_idx], metas_array[0:validation_split_idx],
                              y_array[0:validation_split_idx], args.batch_size)
    test_gen = DataGenerator(embeddings_array[validation_split_idx:], metas_array[validation_split_idx:],
                             y_array[validation_split_idx:], args.batch_size)
except:
    print("Using lite mode")
    train_gen = DataGeneratorLite(embeddings[0:validation_split_idx], metas[0:validation_split_idx],
                                  y[0:validation_split_idx], args.batch_size)
    test_gen = DataGeneratorLite(embeddings[validation_split_idx:], metas[validation_split_idx:],
                                 y[validation_split_idx:], args.batch_size)

def call_existing_code(lr, batch_size):
    pm = k.models.load_model(args.base_model)
    # if we want to replace final CNN layer
    # could have option to retrain entire thing
    if args.from_scratch:  # explicitly state that they are trainable, might be redundant
        for layer in pm.layers:
            layer.trainable = True
    else:
        for layer in pm.layers[0:len(pm.layers) - 3]:
            layer.trainable = False
        for layer in pm.layers[len(pm.layers) - 3:]:
            layer.trainable = True

    pm.compile(optimizer=k.optimizers.Adam(learning_rate=k.optimizers.schedules.ExponentialDecay(
        float(lr), decay_steps=int(validation_split_idx / batch_size), decay_rate=0.9, staircase=False,
        name=None), amsgrad=True), loss='cosine_similarity')
    return pm


def build_model(hp):
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    batch_size = hp.Choice("batch_size", [32, 128, 512])
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(lr, batch_size)
    return model


stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

if args.tuner_search:
    tuner = keras_tuner.Hyperband(build_model, objective="val_loss", max_epochs=args.epochs, directory='hcd_tuning',
                                  project_name='testing', overwrite=True,
                                  hyperband_iterations=args.hyperband_iterations,
                                  distribution_strategy=tf.distribute.MirroredStrategy())
    print(tuner.search_space_summary())

    tuner.search(train_gen, verbose=1, validation_data=test_gen, callbacks=[stop_early])
    print(tuner.results_summary())
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    model = tuner.hypermodel.build(best_hps)
    print("building final model")
    history = model.fit(train_gen, verbose=1, validation_data=test_gen, callbacks=[stop_early], epochs=args.epochs)
    model.save(args.out)
    # model_name = args.out.replace(".h5", "") + "_lr" + args.lrs + "_epoch" + args.epochs + "_batch" + args.batch_size + ".h5"
    # print("saving " + model_name)
    # pm.save(model_name)
else:
    pm = k.models.load_model(args.base_model)
    # if we want to replace final CNN layer
    # could have option to retrain entire thing
    if args.from_scratch:  # explicitly state that they are trainable, might be redundant
        for layer in pm.layers:
            layer.trainable = True
    else:
        for layer in pm.layers[0:len(pm.layers) - 3]:
            layer.trainable = False
        for layer in pm.layers[len(pm.layers) - 3:]:
            layer.trainable = True

    pm.compile(optimizer=k.optimizers.Adam(learning_rate=k.optimizers.schedules.ExponentialDecay(
        float(args.lr), decay_steps=int(validation_split_idx / args.batch_size), decay_rate=0.9, staircase=False,
        name=None), amsgrad=True), loss='cosine_similarity')
    pm.fit(train_gen, epochs=args.epochs, verbose=1, validation_data=test_gen, callbacks=[stop_early])
    pm.save(args.out)
