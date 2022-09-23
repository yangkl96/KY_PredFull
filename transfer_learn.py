#!/usr/bin/env python3

import numpy as np
from pyteomics import mgf
import argparse
import os
import pandas as pd
import sys
import utils

import tensorflow as tf
import tensorflow.keras as k
from tensorflow.keras import backend as K
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
        if index < len(vector): #peaks at 2000 m/z ignored
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
parser.add_argument('--filtered_mgf', type=str, help='mgf to train on', default = "")
parser.add_argument('--mgf_folder', type=str, help='folder with mgf files')
parser.add_argument('--msms_folder', type=str, help='folder with msms identifications') #
parser.add_argument('--base_model', type=str, help='base model to transfer learn on', default="pm.h5")
parser.add_argument('--fragmentation', type=str, help='fragmentation method (HCD, CID, etc)', default = "hcd") #
parser.add_argument('--min_score', type=int, help='minimum Andromeda score', default = 100)
parser.add_argument('--nce', type=int, help='normalized collision energy', default = 30)
parser.add_argument('--epochs', type=int, help='number of epochs to transfer learn model', default = 20)
parser.add_argument('--out', type=str, help='filename to save the transfer learned model', default = "")

args = parser.parse_args()

if args.out == "":
    print("must include transfer learned model name with --out")
    sys.exit()
if args.nce == "":
    print("must include normalized collision energy value with --nce")
    sys.exit()

if args.filtered_mgf == "":
    #preparing annotated mgf files first
    filtered_mgf = []
    for root, dirs, files in os.walk(args.msms_folder):
        for file in files:
            if file == "msms.txt":
                df = pd.read_csv(os.path.join(root, file), sep="\t")
                df = df[df["Score"] >= args.min_score]
                df = df[df["Fragmentation"].str.lower() == args.fragmentation.lower()]

                df_dict = {}
                def get_dict_entry(x):
                    df_dict[x["Scan number"]] = x["Modified sequence"][1:len(x["Modified sequence"]) - 1]#.replace("M(ox)", "m")
                df.apply(lambda x: get_dict_entry(x), axis = 1)

                # read in potential mgf files
                mgf_file = args.mgf_folder + "/" + df["Raw file"].values[0] + ".mgf"
                if not os.path.exists(mgf_file):
                    continue
                print(mgf_file)
                spectra = readmgf(mgf_file)

                for sp in spectra:
                    #check if sp is suitable for inclusion, based on predfull data preprocessing standards, and has ID
                    length = len(sp["mz"])
                    scan_num = int(sp["pep"].split(" ")[0].split(".")[1])
                    if length > 20: #underfragmented
                        if length < 500: #overfragmented
                            if scan_num in df_dict.keys():
                                #can't seem to find entries with >200 ppm difference?
                                sp["params"] = {}
                                sp["params"]["charge"] = sp["charge"]
                                sp["params"]["pepmass"] = sp["mass"]
                                sp["params"]["seq"] = df_dict[scan_num]
                                #MAX_PEPTIDE_LENGTH = np.max(MAX_PEPTIDE_LENGTH, df_dict[scan_num] + 2)
                                sp["params"]["nce"] = args.nce #can try making charge dependent, and add a default
                                sp["m/z array"] = sp["mz"]
                                sp["intensity array"] = sp["it"]
                                filtered_mgf.append(sp)
#            if len(filtered_mgf) > 0:
#                break
    
    mgf.write(filtered_mgf, output = args.mgf_folder + "/filtered.mgf", file_mode = "w")
    args.filtered_mgf = args.mgf_folder + "/filtered.mgf"

K.clear_session()

pm = k.models.load_model(args.base_model)

#adding parts for transfer learning (2 options)

#if we want to replace final CNN layer
for layer in pm.layers[0:len(pm.layers) - 3]:
    layer.trainable = False
    pm.compile(optimizer=k.optimizers.Adam(lr=0.0003), loss='cosine_similarity')
#print(pm.summary())

#if we want to add another CNN layer
#this adds 40 million parameters
#pm = k.models.Model(pm.input, pm.layers[-2].output)
#pm.trainable = False
#output = pm.layers[-2].output
#output = k.layers.Conv1D(SPECTRA_DIMENSION, kernel_size=1, padding='valid', name='conv1d_14')(output)
#output = Activation('sigmoid', name='activation_12')(output)
#output = k.layers.GlobalAveragePooling1D(name='spectrum')(output)
#pm = k.models.Model(pm.input, output)
#print(pm.summary())


print('Reading mgf...')
spectra = readmgf(args.filtered_mgf)

for sp in spectra:
    utils.xshape[0] = max(utils.xshape[0], len(sp["pep"]) + 2)  # update xshape to match max input peptide

#the below is making it seem like there are multiple types of input
y = [spectrum2vector(sp['mz'], sp['it'], sp['mass'], utils.precision, sp['charge']) for sp in spectra]
infos = utils.preprocessor(spectra)
embeddings = [info for info in infos[0]]
metas = [info for info in infos[1]]

y_array = np.zeros((len(y), len(y[0])))
for i in range(len(y)):
    y_array[i] = y[i]
embeddings_array = np.zeros((len(embeddings), embeddings[0].shape[0], embeddings[0].shape[1]))
for i in range(len(embeddings)):
    embeddings_array[i] = embeddings[i]
metas_array = np.zeros((len(metas), metas[0].shape[0], metas[0].shape[1]))
for i in range(len(metas)):
    metas_array[i] = metas[i]

#make a version where instead it makes numpy array from list of flattened numpy arrays
#NOTE: check what shape predict function has it as

#pm.fit(x=asnp32(x), y=asnp32(y), epochs=50, verbose=1)
pm.fit(x=[embeddings_array, metas_array], y=y_array, epochs=args.epochs, verbose=1)
pm.save(args.out)
