# zPolaron
Zelixir's Protein-sequence optimization using Ligand-residue interactions and evolution


## Installation
zPolaron relies on the following packages:

    esm2
    pytorch
    numpy
    PyG
    torch-scatter
    openbabel

## Usage
### 1. For sequence design

(1) Redesign residues 36 and 109 of chain A and residue 126 of chain B, considering interactions with residues within 40 Å of the ligand.

    python run_design.py -r examples/protein.pdb -l examples/ligand.mol2 -a A36,A109,B126 -c 40 -o out/design_out.txt
    
(2) Without specifying the residues to be redesigned, redesign all residues within 40 Å of the ligand based on the current sequence.

    python run_design.py -r examples/protein.pdb -l examples/ligand.mol2 -c 40 -o out/design_out_all.txt
    
### 2. For scoring mutant sequences

Mutate residue 107 of Chain A to T, residue 120 of Chain A to I, and residue 91 of Chain B to Y.

    python run_score.py -r protein.pdb -l ligand.mol2 -c 40 -t A_A107T,A_F120I,B_N91Y -o out/score_out.txt

## Citation
To be updated

## Contacts
To be updated
