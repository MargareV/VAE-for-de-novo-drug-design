import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

print("Keras: %s"%keras.__version__)
print("RDKit: %s"%rdkit.__version__)

data = pd.read_csv('/home/margs/Drug dicovery and machine learning/zinc_image/smiles_zinc.csv', skip_blank_lines=True, nrows=3000, usecols=[0])
data["mol"] = data["smiles"].apply(Chem.MolFromSmiles)
print(data.shape)
#data = np.array(data)
#print(data.shape)

#data = np.reshape(data, (15216,))
#print(data.shape)

#new_data = data[0:3000,]
#print(new_data.shape)

#smile = new_data[0]
#mol = Chem.MolFromSmiles(smile)

def chemcepterize_mol(mol, embed=20.0, res=0.5):
    dims = int(embed*2/res)
    cmol = Chem.Mol(mol.ToBinary())
    cmol.ComputeGasteigerCharges()
    AllChem.Compute2DCoords(cmol)
    coords = cmol.GetConformer(0).GetPositions()
    vect = np.zeros((dims,dims, 1))
    #Bonds first
    for i,bond in enumerate(mol.GetBonds()):
        bondorder = bond.GetBondTypeAsDouble()
        bidx = bond.GetBeginAtomIdx()
        eidx = bond.GetEndAtomIdx()
        bcoords = coords[bidx]
        ecoords = coords[eidx]
        frac = np.linspace(0,1,int(1/res*2)) #
        for f in frac:
            c = (f*bcoords + (1-f)*ecoords)
            idx = int(round((c[0] + embed)/res))
            idy = int(round((c[1]+ embed)/res))
            #Save in the vector first channel
            vect[ idx , idy, 0] = bondorder
    for i,atom in enumerate(cmol.GetAtoms()):
            idx = int(round((coords[i][0] + embed)/res))
            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
            vect[ idx , idy, 0] = atom.GetAtomicNum()
    #Atom Layers
#    for i,atom in enumerate(cmol.GetAtoms()):
#            idx = int(round((coords[i][0] + embed)/res))
#            idy = int(round((coords[i][1]+ embed)/res))
            #Atomic number
#            vect[ idx , idy, 1] = 0
            #Gasteiger Charges
#            charge = atom.GetProp("_GasteigerCharge")
#            vect[ idx , idy, 2] = 0
            #Hybridization
#            hyptype = atom.GetHybridization().real
#            vect[ idx , idy, 2] = hyptype
    return vect


def vectorize(mol):
    return chemcepterize_mol(mol, embed=12)
data["molimage"] = data["mol"].apply(vectorize)
print(data.shape)
#print(smile)
mol = data["mol"][2456]
v = chemcepterize_mol(mol, embed=20, res=0.5)
print(v)
print(v.shape)
v = np.reshape(v, (80, 80))
plt.imshow(v, cmap='gray')
plt.show()