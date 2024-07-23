import copy
import json
import os
import pathlib
import subprocess

import numpy as np
import pytest
from biopandas.pdb import PandasPdb


@pytest.fixture()
def fasta_to_check() -> list:
    """
    Protein ids and the corresponding fasta lengths to check
    """
    _data = [
        {"name": "2e6f_prot_A", "answer": 312, "note": ""},
        {"name": "2leg_prot_A", "answer": 188, "note": "Can be about model selection."},
        {"name": "2leg_prot_B", "answer": 134, "note": "Can be about model selection."},
        {
            "name": "10mh_prot_A",
            "answer": 327,
            "note": "Can be about nucleic acid handling.",
        },
        {
            "name": "1a8j_prot_H",
            "answer": 216,
            "note": "Can be about nonstandard residue handling.",
        },
        {
            "name": "1a8j_prot_L",
            "answer": 216,
            "note": "Can be about nonstandard residue handling.",
        },
        {
            "name": "2a2q_prot_H",
            "answer": 254,
            "note": "Can be about insertion handling.",
        },
    ]
    return _data


@pytest.fixture()
def reordering_to_check() -> list:
    """
    Protein ids and correct residue information to check
    """
    _data = [
        {
            "name": "7eue_prot",
            "answer": {"first_residue_ids": [1, 1], "seq_lens": [119, 111]},
            "note": {"first_residue_ids": "", "seq_lens": ""},
        },
    ]
    return _data


@pytest.fixture()
def resnums_to_check() -> list:
    """
    Protein ids and their corresponding residue numbers to check,
    total and per each chain
    """
    _data = [
        {
            "name": "5dw8_ligand_A_302_2AM",
            "answer": {"total": 20, "A": 17, "B": 3},
            "note": {"total": "", "A": "", "B": ""},
        },
    ]
    return _data


def check_fasta(path_test: str, name: str, answer: int, note: str):
    """
    Checks if sequence lengths are correct

    Parameters
    ----------
    path_test
        test dir root
    name
        protein id
    answer
        correct seq length
    note
        error msg note
    """
    seq_file_path = os.path.join(path_test, "sequences", f"{name}.fasta")
    err_msg = (
        f"Sequence length for {name} does not look correct, "
        f"potential error on protein processing."
    )
    err_msg = " ".join([err_msg, note])
    with open(seq_file_path) as seq_f:
        seq = seq_f.readlines()[1]
    assert len(seq) == answer, err_msg


def check_reordering(path_test: str, name: str, answer: dict, note: dict):
    """
    Checks if chain reordering is correct

    Parameters
    ----------
    path_test
        test dir root
    name
        protein id
    answer
        dictionary of answers per property to check
    note
        dictionary of error msg notes per property to output
    """
    protein_path = os.path.join(path_test, "tmp", name, f"{name}.pdb")
    err_msgs = {
        "first_residue_ids": f"Chains don't start from 1 for {name}, potential error on protein processing.",
        "seq_lens": f"Sequence lenghts for {name} foes not look correct, potential error on protein processing.",
    }
    err_msgs = {k: " ".join([v, note[k]]) for k, v in err_msgs.items()}
    ppdb_orig = PandasPdb().read_pdb(protein_path)
    chains = ppdb_orig.df["ATOM"]["chain_id"].unique().tolist()
    first_residue_ids = []
    seq_lens = []
    for chain in chains:
        ppdb = copy.deepcopy(ppdb_orig)
        ppdb.df["ATOM"] = ppdb.df["ATOM"][ppdb.df["ATOM"]["chain_id"] == chain]
        residue_ids = ppdb.df["ATOM"]["residue_number"].unique().tolist()
        first_residue_ids.append(min(residue_ids))
        seq = list(ppdb.amino3to1()["residue_name"])
        seq_lens.append(len(seq))
    assert first_residue_ids == answer["first_residue_ids"], err_msgs[
        "first_residue_ids"
    ]
    assert seq_lens == answer["seq_lens"], err_msgs["seq_lens"]


def check_resnums(path_test: str, name: str, answer: dict, note: dict):
    """
    Check if residue numbers are consistent for both processed
    and not processed structures

    Parameters
    ----------
    path_test
        test dir root
    name
        protein id
    answer
        dictionary of answers per chain to check
    note
        dictionary of error msg notes per chain to output
    """
    test_states = [
        {
            "fname_tail": "",
            "test_note": "Potential error in pocket residue determination.",
        },
        {
            "fname_tail": "_processed",
            "test_note": "Potential error in pocket residue determination "
            "or protein processing.",
        },
    ]
    for state in test_states:
        fname_tail, test_note = state["fname_tail"], state["test_note"]
        path_residues = os.path.join(
            path_test,
            "pocket_residues",
            f"{name}_neighbouring_residues{fname_tail}.json",
        )
        with open(path_residues) as _f:
            residues = json.load(_f)
        for chain_ in answer:
            answer_, note_ = answer[chain_], note[chain_]
            err_msg = f"Total number of residues for {name} is not correct. "
            err_msg = " ".join([err_msg, test_note, note_])
            if chain_ == "total":
                res = len([one for one in residues if one is not None])
            else:
                res = len(
                    [one for one in residues if one is not None and one[0] == chain_]
                )
            assert res == answer_, err_msg


def check_residue_nones(path_test: str):
    """
    Checks if there are nones in the residue definitions

    Parameters
    ----------
    path_test
        test dir root
    """
    path_pocket_res = os.path.join(path_test, "pocket_residues")
    for pocket_residues_file in os.listdir(path_pocket_res):
        residues = json.load(open(os.path.join(path_pocket_res, pocket_residues_file)))
        for residue in residues:
            if residue is None:
                raise ValueError(
                    "'None' is detected in pocket residues, "
                    "there should be some error in pocket residue determination"
                )


def check_distance_matrices(
    path_test: str, path_ref: str, thresholds: list[float], rtol=1e-05
):
    """
    Checks if the distance matrices are correct

    Parameters
    ----------
    path_test
        test dir root
    path_ref
        reference data root
    thresholds
        clustering thresholds:
            "all" for full distance matrix
            float for the threshold of a reduced distanc matrix
    rtol
        relative tolerance for np.allclose
    """
    for t_ in thresholds:
        fname_tail_ = "" if t_ == "all" else f"_reduced_clusters_thr{t_}.json"
        note_ = (
            "Distance matrix"
            if t_ == "all"
            else f"Reduced distance matrix for thr {t_}"
        )
        path_dm_ref = os.path.join(path_ref, f"distance_matrix{fname_tail_}.npz")
        path_dm_new = os.path.join(
            path_test, "tmp", f"distance_matrix{fname_tail_}.npz"
        )
        ref_distance_matrix = np.load(path_dm_ref)["arr_0"].astype("float32")
        new_distance_matrix = np.load(path_dm_new)["arr_0"].astype("float32")
        if ref_distance_matrix.shape != new_distance_matrix.shape or not np.allclose(
            ref_distance_matrix, new_distance_matrix, rtol=rtol
        ):
            raise ValueError(
                f"{note_} is not correct."
                "Possible error pairwise distance calculation or pocket representations."
            )


def pocket_list_paths_to_absolute(path_pocket_list: str, path_test: str) -> str:
    """
    Copies a pocket list file to the running directory, makes the paths in it absolute

    Parameters
    ----------
    path_pocket_list
        path to the pocket list

    Returns
    -------
    str
        new path
    """

    def paths_to_abs(ln_) -> str:
        return " ".join(map(os.path.abspath, ln_.strip("\n").split(" "))) + "\n"

    path_parent = os.path.join(str(pathlib.Path(__file__).parent.resolve()))
    path_out = os.path.join(path_test, path_pocket_list.split("/")[-1])

    with open(path_pocket_list) as f_pl:
        lns = f_pl.readlines()

    cwd = os.getcwd()
    os.chdir(path_parent)
    new_lns = list(map(paths_to_abs, lns))
    os.chdir(cwd)

    with open(path_out, "w") as f_pl:
        f_pl.write("".join(new_lns))
    return path_out


def test_full_integration(
    tmp_path,
    path_epocs_run_script,
    path_examples,
    path_data,
    esm_parameters_path,
    use_gpu,
    fasta_to_check,
    reordering_to_check,
    resnums_to_check,
):
    """
    Integration test that checks the full pipeline and requires running ESM
    """
    path_test = str(tmp_path)
    path_examples_abs = pocket_list_paths_to_absolute(path_examples, path_test)
    command = f"cd {tmp_path}; python {path_epocs_run_script} -f {path_examples_abs} \
                                            -pp {esm_parameters_path} \
                                            -np 2 \
                                            -cmin 1.6 \
                                            -cmax 2.4 \
                                            -gpu {use_gpu} \
                                            -debug True \
                                            "
    subprocess.run(command, shell=True)

    print("Time for tests...")

    # check if sequence lengths are correct
    for _fasta_data in fasta_to_check:
        check_fasta(path_test=path_test, **_fasta_data)
    print("Sequence lengths are correct. 1/n")

    # check if chain reordering is correct
    for _reordering_data in reordering_to_check:
        check_reordering(path_test=path_test, **_reordering_data)
    print("Chain reordering is correct. 2/n")

    # check if residue numbers are consistent for both processed and not processed structures.
    for _resnum_data in resnums_to_check:
        check_resnums(path_test=path_test, **_resnum_data)
    check_residue_nones(path_test=path_test)
    print("Pocket residue determination is correct. 3/n")

    # check if the distance matrix is correct
    check_distance_matrices(path_test=path_test, path_ref=path_data, thresholds=["all"])
    print("Distance matrix generation is correct. 4/n")

    # check if the reduced distance matrices are correct
    check_distance_matrices(
        path_test=path_test, path_ref=path_data, thresholds=[1.6, 2.4]
    )
    print("Reduced distance matrix generation is correct. 5/n")
