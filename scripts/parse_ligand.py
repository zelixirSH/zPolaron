import numpy as np

class Ligand:
    def __init__(self, file_path=None):
        self.atoms = {}  # {atom_id: (atom_name, x, y, z, atom_type, charge)}
        self.atom_types = [] # ['C.2', 'C.3', 'O.2', ...]
        self.bonds = {}  # {atom_id: [connected_atom_id1, connected_atom_id2, ...]}
        self.aliphatic_carbons = []  # 脂肪碳列表
        self.aromatic_carbons = []   # 芳香碳列表
        self.aromatic_rings = []     # 芳香碳环列表
        self.hbond_donors_OH = []    # 氢键供体OH列表
        self.hbond_donors_NH = []    # 氢键供体NH列表
        self.hbond_acceptors_O = []  # 氢键受体O列表
        self.hbond_acceptors_N = []  # 氢键受体N列表
        self.weak_hbond_donors = {'ali':[], 'aro':[]}  # 弱氢键供体字典
        self.positively_charged_nitrogens = []  # 带正电的氮原子列表
        self.negatively_charged_oxygens = []    # 带负电的氧原子列表
        self.halogens = []           # 卤素原子列表
        self.xyzs = []

        self._parse_mol2(file_path)
        self._classify_atoms()
        self._identify_aromatic_rings()
        
        self.pocket_center = np.mean(np.array(self.xyzs),axis=0)


    def add_atom(self, atom_id, atom_index, atom_name, x, y, z, atom_type, charge):
        self.atoms[atom_id] = {
            'atom_name': atom_name,
            'atom_index': atom_index,
            'xyz': np.array((float(x), float(y), float(z))),
            'atom_type': atom_type,
            'charge': float(charge)
            }  #(atom_name, float(x), float(y), float(z), atom_type, float(charge))
        self.bonds[atom_id] = []


    def add_bond(self, atom_id1, atom_id2):
        self.bonds[atom_id1].append(atom_id2)
        self.bonds[atom_id2].append(atom_id1)


    def _parse_mol2(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

        in_atoms_section = False
        in_bonds_section = False
        atom_idx = 0
        for line in lines:
            line = line.strip()
            if line.startswith("@<TRIPOS>MOLECULE"):
                in_atoms_section = False
                in_bonds_section = False

            elif line.startswith("@<TRIPOS>ATOM"):
                in_atoms_section = True
                in_bonds_section = False

            elif line.startswith("@<TRIPOS>BOND"):
                in_atoms_section = False
                in_bonds_section = True

            elif line.startswith("@<TRIPOS>"):
                in_atoms_section = False
                in_bonds_section = False

            elif in_atoms_section and line:
                parts = line.split()
                atom_id, atom_name, x, y, z, atom_type, charge = \
                    parts[0], parts[1], parts[2], parts[3], parts[4], parts[5], parts[-1]
                self.add_atom(atom_id, atom_idx, atom_name, x, y, z, atom_type, charge)
                self.atom_types.append(atom_type)
                atom_idx += 1
                if not atom_type.startswith('H'):
                    self.xyzs.append((float(x), float(y), float(z)))

            elif in_bonds_section and line:
                parts = line.split()
                bond_id, atom_id1, atom_id2, bond_type = parts[0], parts[1], parts[2], parts[3]
                self.add_bond(atom_id1, atom_id2)


    def _classify_atoms(self):
        for atom_id, atom_data in self.atoms.items():
            atom_type = atom_data['atom_type']
            if atom_type.startswith("C."):
                if atom_type == "C.ar":
                    self.aromatic_carbons.append(atom_id)
                    for neighbor in self.bonds[atom_id]:
                        neighbor_name = self.atoms[neighbor]['atom_name']
                        if "H" in neighbor_name:
                            self.weak_hbond_donors['aro'].append([atom_id, neighbor]) 
                elif atom_type in {"C.3", "C.2", "C.1"}:
                    self.aliphatic_carbons.append(atom_id)
                    for neighbor in self.bonds[atom_id]:
                        neighbor_name = self.atoms[neighbor]['atom_name']
                        if "H" in neighbor_name:
                            self.weak_hbond_donors['ali'].append([atom_id, neighbor]) 
            
            elif atom_type.startswith("N"):
                is_donor = False
                for neighbor in self.bonds[atom_id]:
                    neighbor_name = self.atoms[neighbor]['atom_name']
                    if "H" in neighbor_name:
                        self.hbond_donors_NH.append([atom_id, neighbor])
                        is_donor = True
                if not is_donor:
                    self.hbond_acceptors_N.append(atom_id)
                if atom_data['charge'] > 0:
                    self.positively_charged_nitrogens.append(atom_id)
                    
            elif atom_type.startswith("O"):
                is_donor = False
                for neighbor in self.bonds[atom_id]:
                    neighbor_name = self.atoms[neighbor]['atom_name']
                    if "H" in neighbor_name:
                        self.hbond_donors_OH.append([atom_id, neighbor])
                        is_donor = True
                if not is_donor:
                    self.hbond_acceptors_O.append(atom_id)
                if atom_data['charge'] < 0:
                    self.negatively_charged_oxygens.append(atom_id)
                    
            elif atom_type in ['F', 'Cl', 'Br', 'I']:
                self.halogens.append(atom_id)


    def _identify_aromatic_rings(self):
        visited = set()
        def is_aromatic(atom_id):
            atom_type = self.atoms[atom_id]['atom_type']
            return atom_type in {"C.ar", "N.ar", "O.ar", "S.ar"}

        def dfs(atom_id, path, ring_atoms):
            if atom_id in path:
                ring_start_index = path.index(atom_id)
                ring = path[ring_start_index:]
                if 5 <= len(ring) <= 6 and all(is_aromatic(atom) for atom in ring):
                    if ring not in ring_atoms:
                        ring_atoms.append(ring)
                return True
            if atom_id in visited:
                return False
            visited.add(atom_id)
            path.append(atom_id)

            for neighbor in self.bonds[atom_id]:
                if is_aromatic(neighbor):
                    dfs(neighbor, path, ring_atoms)

            path.pop()
            return False

        ring_atoms = []
        for atom_id in self.atoms:
            if atom_id not in visited and is_aromatic(atom_id):
                dfs(atom_id, [], ring_atoms)

        self.aromatic_rings = ring_atoms
        

# 示例使用
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", type=str, default='ligand.mol2', help='ligand file, mol2 format')
    
    args = parser.parse_args()
    
    lig_mol2_file = args.l

    dataset = '/home/wzh/data/datasets/pdbbind2019/all/'
    pdb = '3i1y'
    
    lig_mol2_file = f'{dataset}/{pdb}/{pdb}_ligand.mol2'

    molecule = Ligand(lig_mol2_file)
        
    print(molecule.aromatic_rings)

