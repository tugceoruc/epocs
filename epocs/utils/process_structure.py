import os

import pandas as pd
from biopandas.pdb import PandasPdb

from pymol import cmd

from ..constants import aa_map, modified_aminoacids, nucleotides
from .pml import UtilsPml

aa = list(aa_map.keys())


def process_protein(protein, tmp_folder, split_chains=True, only_backbone=True):
    """
    Modifies the pdb files using pymol:
    selects alternative locations, fixes insertions
    and reorders residue ids. Residue order is
    important for sequence and pdb residue matching.

    Modified pdb file is written to the ./tmp file.

    """
    output_folder = tmp_folder + protein.split("/")[-1].split(".")[0]

    os.mkdir(output_folder)
    os.chdir(output_folder)

    cmd.feedback("disable", "all", "actions")
    cmd.feedback("disable", "all", "results")

    # enforce pymol to keep original order of atoms & residues
    # to ensure that renumbering of the residues is consistent
    cmd.set("retain_order", 1)

    cmd.delete("all")
    cmd.load(protein, "prot")

    # fix insertion codes
    # UtilsPml.fix_insertion_codes(sele="prot", logging_name=protein)

    # fix alt locations
    UtilsPml.fix_alt_locs(
        sele="prot",
        logging_name=protein,
        rename_alt_locs=True,
        resnames_to_check=[list(modified_aminoacids.keys()) + nucleotides + aa],
    )

    # fix bfactor, it is set to the same number to keep the columns in order
    # not sure that is necessary anymore
    cmd.alter("prot", "b=20")

    modified_file_name = f'{output_folder}/{protein.split("/")[-1].split(".")[0]}.pdb'
    cmd.save(modified_file_name, "prot")
    cmd.delete("all")

    os.chdir(tmp_folder)

    ppdb = PandasPdb().read_pdb(modified_file_name)

    # select backbone, change modified aa to standard, select model

    ppdb = ppdb.label_models()
    ppdb.df["ATOM"] = ppdb.df["ATOM"][ppdb.df["ATOM"]["model_id"] == 1]

    for (chain_id, res_num, res_name), residue in ppdb.df["HETATM"].groupby(
        ["chain_id", "residue_number", "residue_name"]
    ):

        if res_name in modified_aminoacids:
            residue_mask = (ppdb.df["HETATM"]["chain_id"] == chain_id) & (
                ppdb.df["HETATM"]["residue_number"] == res_num
            )
            ppdb.df["HETATM"].loc[residue_mask, "residue_name"] = modified_aminoacids[
                res_name
            ]
            residue_data = ppdb.df["HETATM"][residue_mask]
            residue_data["record_name"] = "ATOM"
            if modified_aminoacids[res_name] != "XXX":
                ppdb.df["ATOM"] = pd.concat([ppdb.df["ATOM"], residue_data])
            ppdb.df["HETATM"] = ppdb.df["HETATM"][~residue_mask]

    df = ppdb.df["ATOM"]
    filtered_df = df[~df["residue_name"].isin(nucleotides)]
    ppdb.df["ATOM"] = filtered_df
    df = ppdb.df["HETATM"]
    filtered_df = df[~df["residue_name"].isin(nucleotides)]
    ppdb.df["HETATM"] = filtered_df

    ppdb.df["HETATM"] = ppdb.df["HETATM"][ppdb.df["HETATM"]["model_id"] == 1]
    ppdb.df["HETATM"] = ppdb.df["HETATM"][ppdb.df["HETATM"]["element_symbol"] != "H"]
    ppdb.df["HETATM"] = ppdb.df["HETATM"][ppdb.df["HETATM"]["residue_name"] != "HOH"]

    ppdb.df["HETATM"]["b_factor"] = 20

    _atm_m_id_vals = ppdb.df["ATOM"].pop("model_id")  # noqa: F841
    _hetatm_m_id_vals = ppdb.df["HETATM"].pop("model_id")  # noqa: F841

    ppdb.df["ATOM"] = ppdb.df["ATOM"][ppdb.df["ATOM"]["element_symbol"] != "H"]
    if only_backbone:
        ppdb.df["ATOM"] = ppdb.df["ATOM"][
            (ppdb.df["ATOM"]["atom_name"] == "C")
            | (ppdb.df["ATOM"]["atom_name"] == "CA")
            | (ppdb.df["ATOM"]["atom_name"] == "O")
            | (ppdb.df["ATOM"]["atom_name"] == "N")
        ]

    # renumber residues
    atom_df = ppdb.df["ATOM"]

    residue_mapping_tmp = {}
    residue_mapping = {}
    prev_chainID = []
    for (chain_id, res_num, insertion, res_name), group in atom_df.groupby(
        ["chain_id", "residue_number", "insertion", "residue_name"]
    ):
        if prev_chainID != chain_id:
            new_residue_number = 1
        prev_chainID = chain_id
        residue_mapping_tmp[(chain_id, res_num, insertion)] = new_residue_number
        residue_mapping[f"{chain_id}_{res_num}_{res_name}"] = (
            f"{chain_id}_{new_residue_number}_{res_name}"
        )

        new_residue_number += 1

    def map_residue_number(row):
        key = (row["chain_id"], row["residue_number"], row["insertion"])
        return residue_mapping_tmp[key]

    atom_df["residue_number"] = atom_df.apply(map_residue_number, axis=1)
    ppdb.df["ATOM"] = atom_df

    # save modified structure as pdb
    ppdb.to_pdb(path=modified_file_name, records=None, gz=False, append_newline=True)
    return (
        modified_file_name,
        residue_mapping,
        {v: k for k, v in residue_mapping.items()},
    )
