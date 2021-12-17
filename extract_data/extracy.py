from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool

def filt_data(line):
    '''
    Extract small organic molecules
    Rules
    1. Molecules don't have any metallic element and parital charge.
    2. 6 < The number of atoms in each molecule < 50
    3. SMILES must have '3' ward to extract conjugated system.
    '''

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
