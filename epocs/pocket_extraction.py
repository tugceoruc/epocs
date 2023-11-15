import copy
import json
import os
import pickle

from multiprocessing import Pool, Process
from os import listdir
from typing import List

import numpy as np
import pandas as pd
import torch
import tqdm

from biopandas.pdb import PandasPdb
from pymol import cmd
from scipy.spatial import Voronoi
from scipy.spatial.distance import cdist

from .constants import metalions
from .scripts.esm_extract import generate_esm_embeddings
from .utils import process_structure

# to get rid of the biopandas errors with copy&view assignment
# be careful not to get a real error here!!!
pd.options.mode.chained_assignment = None


class GetPockets:
    def __init__(
        self,
        esm_parameters_path: str = "./",
        only_backbone_atoms: bool = True,
        pocket_distance_cutoff: float = 8.0,
        tmp_folder: str = "./tmp/",
        sequences_path: str = "./sequences/",
        embeddings_path: str = "./embeddings/",
        pocket_embeddings_path: str = "./pocket_embeddings/",
        pocket_residues_path: str = "./pocket_residues/",
        use_gpu: bool = True,
        debugging_mode: bool = False,
    ):
        self.esm_parameters_path = esm_parameters_path
        self.dbpath = "/data/databases/pdb/biounit/"
        self.only_backbone_atoms = only_backbone_atoms
        self.pocket_distance_cutoff = pocket_distance_cutoff
        self.tmp_folder = tmp_folder
        self.embeddings_path = embeddings_path
        self.pocket_embeddings_path = pocket_embeddings_path
        self.sequences_path = sequences_path
        self.pocket_residues_path = pocket_residues_path
        self.use_gpu = use_gpu
        self.debugging_mode = debugging_mode

        self.metalions = list(metalions.keys())

        os.system(f"rm -r {self.tmp_folder}")
        os.system(f"mkdir {self.tmp_folder}")

    def isNeighbouringResiude(self, res_coords, ligand_coords, threshold: float = 8.0):
        """
        Checks whether coords in res_coords and
        ligands_coords are closer than a given threshold.

        Both res_coords and ligand_coords can be a set of coordinates or
        an atom coordinate.

        """
        dist = cdist(res_coords, ligand_coords)
        if any(item < threshold for item in dist.flatten()):
            return True
        else:
            return False

    def get_voronoi_neighbours(
        self,
        ppdb: object,
        ligand_coords,
        threshold: bool = None,
        only_backbone: bool = False,
    ):
        """
        Returns the list of neightbouring residues determined by voronoi tessellation.

        If threshold is not set to None, the neighbouring residues are filtered by
        their distance to the ligand unit. For this filtering, if the user wants to use
        only backbone coordinates, they should set only_backbone to True.

        For example, if threshold is set to 5 and only_backbone is set to False,
        for the filtering, the function will look for any atom (backbone + side chain) to be closer
        than 5 A.

        For example, if threhold is set to 8 and only_backbone is set to True,
        for the filtering, the function will look for backbone atoms to be closer
        than 8 A.

        """
        coords = np.array(ppdb.df["ATOM"][["x_coord", "y_coord", "z_coord"]])
        ppdb.df["HETATM"] = ppdb.df["HETATM"][
            ppdb.df["HETATM"]["element_symbol"] != "H"
        ]
        prot_len = len(coords)
        coords = np.vstack([coords, ligand_coords])

        ligand_index = list(range(prot_len, len(coords)))

        vor = Voronoi(coords)
        ridge_points = vor.ridge_points

        neighbour_indices = []

        ligand_index_set = set(ligand_index)

        for i, j in ridge_points:
            if i in ligand_index_set and j not in ligand_index_set:
                neighbour_indices.append(j)
            elif j in ligand_index_set and i not in ligand_index_set:
                neighbour_indices.append(i)
        neighbour_residues = []

        ppdb.df["ATOM"]["line_idx"] = np.arange(ppdb.df["ATOM"].shape[0])

        for i in neighbour_indices:
            selected_res_df = ppdb.df["ATOM"][ppdb.df["ATOM"]["line_idx"] == i]

            res_id = "_".join(
                [
                    selected_res_df["chain_id"].tolist()[0],
                    str(selected_res_df["residue_number"].tolist()[0]),
                    selected_res_df["residue_name"].tolist()[0],
                ]
            )
            if res_id not in neighbour_residues:
                neighbour_residues.append(res_id)
        if threshold is not None:
            if only_backbone:
                ppdb.df["ATOM"] = ppdb.df["ATOM"][
                    (ppdb.df["ATOM"]["atom_name"] == "C")
                    | (ppdb.df["ATOM"]["atom_name"] == "CA")
                    | (ppdb.df["ATOM"]["atom_name"] == "O")
                    | (ppdb.df["ATOM"]["atom_name"] == "N")
                ]

            ligand_coords = coords[ligand_index, :]
            close_neighbour_residues = []
            for residue in neighbour_residues:
                chain_id, residue_number, residue_name = residue.split("_")
                residue_coord = ppdb.df["ATOM"][
                    (ppdb.df["ATOM"]["chain_id"] == chain_id)
                    & (ppdb.df["ATOM"]["residue_number"] == int(residue_number))
                    & (ppdb.df["ATOM"]["residue_name"] == residue_name)
                ][["x_coord", "y_coord", "z_coord"]]
                if self.isNeighbouringResiude(
                    residue_coord, ligand_coords, threshold=threshold
                ):
                    close_neighbour_residues.append(residue)
            neighbour_residues = close_neighbour_residues
        if len(neighbour_residues) > 0:
            return neighbour_residues
        else:
            return None

    def intersection(self, lst1: List, lst2: List):
        """
        Compares two lists and returns the mutual elements in both lists.
        """
        return [value for value in lst1 if value in lst2]

    def extract_sort_key(self, s: List):
        """
        Sorts the residues based on their chain ID and residue number.
        """
        chain = s.split("_")[0]
        resNo = int(s.split("_")[1])
        return (chain, resNo)

    def get_esm_embeddings(self, neighbouring_residues: List, name: str):
        """
        Returns average of neighbour embeddings.

        neighbouring_residues: list of the neighbouring residues in the format of chainID_residueID_residueName
        name: name of esm file
        """
        neighbouring_residues = sorted(neighbouring_residues, key=self.extract_sort_key)
        neighbouring_residues = [
            one for one in neighbouring_residues if one.split("_")[2] != "UNK"
        ]
        chains = np.unique(
            [
                neighbour_residue.split("_")[0]
                for neighbour_residue in neighbouring_residues
            ]
        )
        neighbour_embeddings = []
        for chainID in chains:
            pt_path = f"{self.embeddings_path}/{name}_{chainID}.pt"
            if os.path.exists(pt_path):
                esm_model = torch.load(pt_path)
                for idx, res in enumerate(neighbouring_residues):
                    if res.split("_")[0] == chainID:
                        resno = int(res.split("_")[1]) - 1
                        if resno < len(esm_model["representations"][36]):
                            neighbour_embeddings.append(
                                esm_model["representations"][36][resno, :]
                                .detach()
                                .cpu()
                                .numpy()
                            )
                        else:
                            expected_model_no = int(
                                np.floor(resno / len(esm_model["representations"][36]))
                            )
                            resno = (
                                resno
                                - len(esm_model["representations"][36])
                                * expected_model_no
                            )
                            neighbour_embeddings.append(
                                esm_model["representations"][36][resno, :]
                                .detach()
                                .cpu()
                                .numpy()
                            )

            else:
                print(
                    f"{pt_path} does not exist. Check the residue embeddings for chain {chainID} for {name}."
                )
                return []

        if len(neighbour_embeddings) == len(neighbouring_residues):
            return np.mean(
                np.array(neighbour_embeddings).reshape(
                    (
                        -1,
                        len(
                            esm_model["representations"][36][0, :]
                            .detach()
                            .cpu()
                            .numpy()
                        ),
                    )
                ),
                axis=0,
            )
        else:
            print(f"Problem with protein {name}")
            return []

    def save_neighbours(
        self, neighbour_list, residue_mapping_rev, pocket_name, neighbour_type
    ):
        """
        Takes the pocket residue list and saves the residues
        in both processed and unprocessed residue labels in
        <chain_id>_<residue_id>_<residue_name> format.
        """

        out_name = (
            f"{self.pocket_residues_path}/{pocket_name}_{neighbour_type}_processed.json"
        )
        with open(out_name, "w") as mfile:
            json.dump(neighbour_list, mfile)

        unprocessed_neighbour_list = list(map(residue_mapping_rev.get, neighbour_list))

        out_name = f"{self.pocket_residues_path}/{pocket_name}_{neighbour_type}.json"
        with open(out_name, "w") as mfile:
            json.dump(unprocessed_neighbour_list, mfile)

    def get_coordinates(self, ligand):
        """
        Gets pymol ligand object and returns coordinates of ligand atoms.
        """

        cmd.load(ligand, "ligand_structure")
        structure = cmd.get_model("ligand_structure")
        coords = []
        for atom in structure.atom:
            if atom.symbol != "H":
                coords.append(atom.coord)
        cmd.delete("ligand_structure")

        return np.vstack(coords)

    def extract_sequence(self, protein):
        """
        Gets protein path and saves the sequences of protein chains.

        protein: path to protein.
        """

        if not protein.endswith("pdb") and not protein.endswith("cif"):
            raise ValueError("Protein input file format should be pdb or cif.")

        protein_path, _, _ = process_structure.process_protein(
            protein=protein,
            tmp_folder=self.tmp_folder,
            split_chains=True,
            only_backbone=True,
        )
        ppdb_orig = PandasPdb().read_pdb(protein_path)
        chains = ppdb_orig.df["ATOM"]["chain_id"].unique().tolist()
        for chain in chains:
            ppdb = copy.deepcopy(ppdb_orig)
            ppdb.df["ATOM"] = ppdb.df["ATOM"][ppdb.df["ATOM"]["chain_id"] == chain]
            seq = ppdb.amino3to1()
            seq = "".join(list(seq["residue_name"]))

            if len(seq) > 10 and len(seq) < 1022:
                name = f"{protein.split('/')[-1].split('.')[0]}"
                out = open(f"{self.sequences_path}{name}_{chain}.fasta", "w")
                out.write(f">{name}_{chain}\n")
                out.write(seq)
                out.close()
        command = (
            f"rm -r {'/'.join([folder for folder in protein_path.split('/')[:-1]])}"
        )
        os.system(command)

    def extract_pocket(self, protein, ligands):
        """
        Gets the paths of protein and corresponding ligand(s),
        determines the pocket residues, generates pocket representations from esm residue embeddings,
        and saves the pocket representations and pocket residues.
        """

        (protein_path, residue_mapping, residue_mapping_rev) = (
            process_structure.process_protein(
                protein=protein,
                tmp_folder=self.tmp_folder,
                split_chains=True,
                only_backbone=True,
            )
        )
        ppdb = PandasPdb().read_pdb(protein_path)
        for ligand in ligands:
            if ligand.endswith("json"):
                ligand_coords = json.load(open(ligand))
                threshold = 10
            else:
                ligand_coords = self.get_coordinates(ligand)
                threshold = self.pocket_distance_cutoff

            neighbouring_residues = self.get_voronoi_neighbours(
                ppdb,
                ligand_coords,
                threshold=threshold,
                only_backbone=self.only_backbone_atoms,
            )

            if neighbouring_residues:
                embedding = self.get_esm_embeddings(
                    neighbouring_residues, protein.split("/")[-1].split(".")[0]
                )

                if len(embedding) > 0:
                    out_pocket_name = ligand.split("/")[-1].split(".")[0]
                    outname = f"{self.pocket_embeddings_path}/{out_pocket_name}.pkl"
                    pickle.dump(embedding, open(outname, "wb"))
                    self.save_neighbours(
                        neighbouring_residues,
                        residue_mapping_rev,
                        pocket_name=out_pocket_name,
                        neighbour_type="neighbouring_residues",
                    )
                else:
                    print(
                        f"Pocket embedding could not be generated for ligand {ligand}"
                    )
            else:
                print(f"No pocket residue was detected for ligand {ligand}")

        command = (
            f"rm -r {'/'.join([folder for folder in protein_path.split('/')[:-1]])}"
        )
        if not self.debugging_mode:
            os.system(command)

    def run_esm(self):
        """
        Runs ESM protein language model on chain sequences
        and saves esm residue embeddings.
        """

        print("ESM embeddings are about to be generated...")
        seqs = {}
        cntr = 0
        for seqf in listdir(self.sequences_path):
            f = open(self.sequences_path + seqf)
            seq_fmt = f.readlines()
            f.close()
            if "?" not in seq_fmt[1]:
                seqs[cntr] = [seq_fmt[0], seq_fmt[1]]
                cntr += 1
            else:
                print(f"Sequence contains unaccpeted residue for sequence {seq_fmt[0]}")

        cntr = 0
        size = 1000
        group_no = int(np.ceil(len(seqs) / size))
        for g in range(group_no):

            out = open(f"{self.sequences_path}all_group_{g}.fasta", "w")
            for seqno in seqs:
                if seqno < (g + 1) * size and seqno >= g * size:
                    out.write(seqs[seqno][0])
                    out.write(seqs[seqno][1] + "\n")
            out.close()
            args = (
                self.esm_parameters_path,
                f"{self.sequences_path}all_group_{g}.fasta",
                self.embeddings_path,
                "per_tok",
                self.use_gpu,
            )
            process = Process(target=generate_esm_embeddings, args=args)
            process.start()
            process.join()

        print("ESM embeddings are generated.")

    def run_sequence_extraction(self, protein):
        self.extract_sequence(protein)

    def run_sequence_extraction_in_parallel(self, items, num_processes=8):
        items = np.unique(list(items.keys())).tolist()

        if num_processes == 1:
            for item in tqdm.tqdm(items):
                self.run_sequence_extraction(item)
        else:
            with Pool(processes=num_processes) as pool:
                list(
                    tqdm.tqdm(
                        pool.imap(self.run_sequence_extraction, items), total=len(items)
                    )
                )

    def run_pocket_extraction(self, item):
        (protein, ligands) = item
        self.extract_pocket(protein, ligands)

    def run_pocket_extraction_in_parallel(self, items, num_processes=8):
        print("Pocket extraction is starting...")
        generated_pockets = os.listdir(self.pocket_embeddings_path)
        items = [
            (one, items[one]) for one in items if f"{one}.pkl" not in generated_pockets
        ]
        if num_processes == 1:
            for item in tqdm.tqdm(items):
                self.run_pocket_extraction(item)
        else:
            with Pool(processes=num_processes) as pool:
                list(
                    tqdm.tqdm(
                        pool.imap(self.run_pocket_extraction, items), total=len(items)
                    )
                )
