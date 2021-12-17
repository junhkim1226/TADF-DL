from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool
import os
import random
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

fdefName = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

def filt_data(line):
    except_list = ['F', 'Cl', 'Br','I','U', 'Te', 'Se', 'Tl', 'Tc','+','.','P=O','P(=O)', 'S=O','S(=O)','Pb','Mn', '-]','A','G','Hg','Sn','Bi','Sb','Nb','V','Cr','W','Co','Pt','Os', 'Mo', 'Rh', 'Ta', 'Ni', 'Re', 'Pd', 'Zn', 'Ru', 'Zr', 'Po', 'Xe', 'Y', 'Cu', 'Sr', 'Pr','*']

    try:
        label = line.split()[1]
        except_val = 0
        for element in except_list:
            if element in label:
                except_val = 1
        if except_val == 0 and '3' in label:
            mol = Chem.MolFromSmiles(label)
            atomnum = mol.GetNumAtoms()
            if atomnum < 50 and atomnum > 6:
                suppl = Chem.ResonanceMolSupplier(mol)
                num_conj = suppl.GetNumConjGrps()
                if num_conj > 0:
                    return line
    except:
        a = 1
    return 0

if __name__ =='__main__':
    #Read pubchem data
    f = open('./pubchem.txt','r')
    lines = f.readlines()
    f.close()

    #Extract small organic molecules
    pool = Pool(processes = 28)
    results = pool.map(filt_data,lines)
    pool.close()
    pool.join()

    #Write extracted molecules
    f = open('screened_molecule.txt','w')
    for result in results:
        if result != 0:
            f.write(result)
    f.close()

    random.seed(0)
    inp = 'screened_molecule.txt'
    num_data = 50000

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
