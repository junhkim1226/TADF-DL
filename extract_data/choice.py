import os
import random
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from rdkit import Chem

fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

#### Option
random.seed(0)
inp = 'screened_molecule.txt'
num_data = 50000

#### Main

f = open(inp,'r')
lines = f.readlines()
f.close()

sample_list = random.sample(lines,num_data)
f = open('sampled_data.txt','w')
for line in sample_list:
    cnt = 0

    CID,smiles = line.strip().split()

    mol = Chem.MolFromSmiles(smiles)
    feats = factory.GetFeaturesForMol(mol)

    for i in range(len(feats)):
        if feats[i].GetFamily() == 'Aromatic':
            cnt += 1
    if cnt > 0:
        f.write(line)
f.close()
