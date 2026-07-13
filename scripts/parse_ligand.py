import numpy as np
from collections import defaultdict

class Ligand:
    def __init__(self, file_path=None):
        self.atoms = {}  # {atom_id: {...}}
        self.atom_types = []  # ['C.2', 'C.3', ...]
        self.bonds = {}  # {atom_id: [connected_atom_id1, ...]}

        self.aliphatic_carbons = []
        self.aromatic_carbons = []
        self.aromatic_rings = []

        self.hbond_donors_OH = []
        self.hbond_donors_NH = []
        self.hbond_acceptors_O = []
        self.hbond_acceptors_N = []

        self.weak_hbond_donors = {'ali': [], 'aro': []}

        self.positively_charged_nitrogens = []
        self.negatively_charged_oxygens = []
        self.halogens = []

        self.xyzs = []

        self._parse_mol2(file_path)
        self._classify_atoms()
        self._identify_aromatic_rings()

        if self.xyzs:
            self.pocket_center = np.mean(np.array(self.xyzs), axis=0)
        else:
            self.pocket_center = np.array([0.0, 0.0, 0.0])

    def add_atom(self, atom_id, atom_index, atom_name, x, y, z, atom_type, charge):
        self.atoms[atom_id] = {
            'atom_name': atom_name,
            'atom_index': atom_index,
            'xyz': np.array((float(x), float(y), float(z))),
            'atom_type': atom_type,
            'charge': float(charge)
        }
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
                atom_id = int(parts[0])
                atom_name = parts[1]
                x, y, z = parts[2], parts[3], parts[4]
                atom_type = parts[5]
                charge = parts[-1]

                self.add_atom(atom_id, atom_idx, atom_name, x, y, z, atom_type, charge)
                self.atom_types.append(atom_type)

                atom_idx += 1

                if atom_type != 'H':
                    self.xyzs.append((float(x), float(y), float(z)))

            elif in_bonds_section and line:
                parts = line.split()
                atom_id1 = int(parts[1])
                atom_id2 = int(parts[2])
                self.add_bond(atom_id1, atom_id2)

    def _classify_atoms(self):
        for atom_id, atom_data in self.atoms.items():
            atom_type = atom_data['atom_type']

            if atom_type.startswith("C."):
                if atom_type == "C.ar":
                    self.aromatic_carbons.append(atom_id)
                    for neighbor in self.bonds.get(atom_id, []):
                        if self.atoms[neighbor]['atom_type'] == 'H':
                            self.weak_hbond_donors['aro'].append([atom_id, neighbor])

                elif atom_type in {"C.3", "C.2", "C.1"}:
                    self.aliphatic_carbons.append(atom_id)
                    for neighbor in self.bonds.get(atom_id, []):
                        if self.atoms[neighbor]['atom_type'] == 'H':
                            self.weak_hbond_donors['ali'].append([atom_id, neighbor])

            elif atom_type.startswith("N"):
                is_donor = False
                for neighbor in self.bonds.get(atom_id, []):
                    if self.atoms[neighbor]['atom_type'] == 'H':
                        self.hbond_donors_NH.append([atom_id, neighbor])
                        is_donor = True
                if not is_donor:
                    self.hbond_acceptors_N.append(atom_id)

                if atom_data['charge'] > 0:
                    self.positively_charged_nitrogens.append(atom_id)

            elif atom_type.startswith("O"):
                is_donor = False
                for neighbor in self.bonds.get(atom_id, []):
                    if self.atoms[neighbor]['atom_type'] == 'H':
                        self.hbond_donors_OH.append([atom_id, neighbor])
                        is_donor = True
                if not is_donor:
                    self.hbond_acceptors_O.append(atom_id)

                if atom_data['charge'] < 0:
                    self.negatively_charged_oxygens.append(atom_id)

            elif atom_type in ['F', 'Cl', 'Br', 'I']:
                self.halogens.append(atom_id)

    def _identify_aromatic_rings(self):
        atoms = self.atoms
        bonds = self.bonds

        def is_aromatic(atom_id):
            return atoms[atom_id]['atom_type'] in {"C.ar", "N.ar", "O.ar", "S.ar"}

        aromatic_atoms = [atom_id for atom_id in atoms if is_aromatic(atom_id)]
        aromatic_set = set(aromatic_atoms)

        if not aromatic_atoms:
            self.aromatic_rings = []
            return

        found_signatures = set()
        ring_atoms = []

        def dfs(start_atom, current_atom, path):
            if len(path) > 6:
                return

            for neighbor in bonds.get(current_atom, []):
                if neighbor == start_atom and len(path) in (5, 6):
                    signature = frozenset(path)
                    if signature not in found_signatures:
                        found_signatures.add(signature)
                        ring_atoms.append(sorted(signature))
                    continue

                if neighbor in aromatic_set and neighbor not in path:
                    dfs(start_atom, neighbor, path + [neighbor])

        for atom in aromatic_atoms:
            dfs(atom, atom, [atom])

        self.aromatic_rings = ring_atoms


if __name__ == "__main__":
    dataset = '/home/wzh/data/datasets/pdbbind2019/all/'
    pdb = '3i1y'
    lig_mol2_file = f'{dataset}/{pdb}/{pdb}_ligand.mol2'

    molecule = Ligand(lig_mol2_file)
    print(molecule.aromatic_rings)
