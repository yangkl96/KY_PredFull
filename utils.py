import numpy as np
import math
from pyteomics import mass
import tensorflow.keras as k

# Hyper Parameters
precision = 0.1
low = 0
dim = 20000
upper = math.floor(low + dim * precision)
max_mz = dim * precision + low

max_out = dim
meta_shape = (3, 30) # (charge, ftype, other(mass, nce))
max_charge = 30

mz_scale = 20000.0
len_scale = 1000

extra_ptm_slots = 5

mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538}

#needs some work
Alist = list('ACDEFGHIKLMNPQRSTVWYZ')
oh_dim = len(Alist) + 3
#x_dim = oh_dim + 2 + 3
xshape = [1, oh_dim + extra_ptm_slots]

# fragmentation types
types = {'un': 0, 'cid': 1, 'etd': 2, 'hcd': 3, 'ethcd': 4, 'etcid': 5}

charMap = {'*': 0, ']': len(Alist) + 1, '[': len(Alist) + 2} #even though we don't encounter this, keep because predfull was trained this way
for i, a in enumerate(Alist): charMap[a] = i + 1

# help function to parse modifications

def getmod(pep): #might be smart to encode every mod by its PTM mass
    mod = np.zeros(len(pep)) #encoded either as 1 for oxM, -2 for other mod, or float for generic PTM mass

    if pep.isalpha(): return pep, mod, 0

    seq = []
    nmod = 0

    i = -1
    while len(pep) > 0:
        if pep[0] == '(':
            if pep[:3] == '(O)':
                mod[i] = 1
                pep = pep[3:]
            elif pep[:4] == '(ox)':
                mod[i] = 1
                pep = pep[4:]
            #elif pep[2] == ')' and pep[1] in 'ASDFGHJKLZXCVBNMQWERTYUIOP': #use form (aa) to symbolize PTM
            #    mod[i] = -2 #because of how fastmass works, comment out for now
            #    pep = pep[3:]
            else:
                raise 'unknown mod: ' + pep

        elif pep[0] == '+' or pep[0] == '-': #write PTM mass in mod list
            sign = 1 if pep[0] == '+' else -1

            for j in range(1, len(pep)):
                if pep[j] not in '.1234567890':
                    if i == -1: #N-term mod
                        nmod += sign * float(pep[1:j])
                    else:
                        mod[i] += sign * float(pep[1:j])
                    pep = pep[j:]
                    break
                if j == len(pep) - 1:# and pep[-1] in '.1234567890': # till end
                    mod[i] += sign * float(pep[1:])
                    pep = ""
                    break
        else:
            seq += pep[0]
            pep = pep[1:]
            i = len(seq) - 1 # more realible

    return ''.join(seq), mod[:len(seq)], nmod

# help functions
def mz2pos(mz, pre=precision): return int(round((mz - low) / pre))


def pos2mz(pos, pre=precision): return pos * pre + low


def asnp(x): return np.asarray(x)


def asnp32(x): return np.asarray(x, dtype='float32')


def f2(x): return "{0:.2f}".format(x)


def f4(x): return "{0:.4f}".format(x)


# compute percursor mass
def fastmass(pep, ion_type, charge, mod=None, cam=True):
    base = mass.fast_mass(pep, ion_type=ion_type, charge=charge)

    if cam:
        base += 57.021 * pep.count('C') / charge

    if not mod is None:
        base += 15.995 * np.sum(mod == 1) / charge

        base += -np.sum(mod[mod < 0])
    return base


# embed input item into a matrix
def embed(spectrum, mass_scale=200, embedding=None, pep=None):  # changed mass scale from 2000
    if 'mod' not in spectrum:
        pep, mod, _ = getmod(spectrum["pep"])
        spectrum["mod"] = mod
        spectrum["pep"] = pep
    if pep is None: pep = spectrum['pep']

    # pep = pep.replace('L', 'I') #Kevin: why would you replace this?
    if embedding is None:
        embedding = np.zeros(xshape, dtype='float32')

    embedding[len(pep)][oh_dim - 1] = 1  # ending pos [
    for i, aa in enumerate(pep):
        embedding[i][charMap[aa]] = 1  # 1 - 20
        embedding[i][oh_dim] = mono[aa] / mass_scale

    embedding[:len(pep), oh_dim + 1] = np.arange(len(pep)) / len_scale  # position info
    embedding[len(pep) + 1, 0] = 1  # padding info *

    for i, modi in enumerate(spectrum['mod']):
        embedding[i][oh_dim + 2 + int(modi)] = 1

    return embedding #order of embedding is 0: padding, 1-20: 20 aa, 23: ending, 24: monoisotopic mass, 25: position info, 26:unmodified, 27: oxM

def make_meta(sp, meta=np.zeros(meta_shape, dtype='float32')):
    pep = sp['pep']
    meta[0][sp['charge'] - 1] = 1  # charge
    meta[1][sp['type']] = 1  # ftype
    meta[2][0] = fastmass(pep, ion_type='M', charge=1) / mz_scale

    if not 'nce' in sp or sp['nce'] == 0:
        meta[2][-1] = 0.25
    else:
        meta[2][-1] = sp['nce'] / 100.0
    return meta

# preprocess function for inputs
def preprocessor(batch):
    batch_size = len(batch)
    embedding = np.zeros((batch_size, *xshape), dtype='float32')
    meta = np.zeros((batch_size, *meta_shape), dtype='float32')

    if type(batch) == list:
        for i, sp in enumerate(batch):
            embed(sp, embedding=embedding[i])
            make_meta(sp, meta=meta[i])
    #else:
    #    embed(batch, embedding=embedding[0])
    #    make_meta(batch, meta=meta[0])

    return (embedding, meta)


# generator for inputs
class input_generator(k.utils.Sequence):
    def __init__(self, spectra, processor, batch_size, shuffle=1):
        self.spectra = spectra
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_begin(self, epoch):
        if epoch > 0 and self.shuffle:
            np.random.shuffle(self.spectra)

    def __len__(self):
        return math.ceil(len(self.spectra) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.spectra))

        return (self.processor(self.spectra[start_idx: end_idx]),)


# functions that transfer predictions into mgf format
def sparse(x, y, th=0.0002):
    x = np.asarray(x, dtype='float32')
    y = np.asarray(y, dtype='float32')

    y /= np.max(y)

    return x[y > th], y[y > th]


def tomgf(sp, y):
    head = ("BEGIN IONS\n"
            f"TITLE={sp['title']}\n"
            f"PEPTIDE={sp['title']}\n"
            f"CHARGE={sp['charge']}+\n"
            f"PEPMASS={sp['mass']}\n")

    y[min(math.ceil(sp['mass'] * sp['charge'] / precision), len(y)):] = 0

    imz = np.arange(0, dim, dtype='int32') * precision + low  # more acurate
    mzs, its = sparse(imz, y)

    # mzs *= 1.00052

    peaks = [f"{f2(mz)} {f4(it * 1000)}" for mz, it in zip(mzs, its)]

    return head + '\n'.join(peaks) + '\nEND IONS'