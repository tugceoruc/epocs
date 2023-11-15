#! /usr/bin/env python
import argparse
import os

from epocs.clustering import ClusterPockets
from epocs.pocket_extraction import GetPockets


def make_argument_parser():
    parser = argparse.ArgumentParser(description="EPoCS-based pocket clustering tool")
    parser.add_argument("-f", "--file", help="list of proteins and ligands")
    parser.add_argument(
        "-pp",
        "--esm_parameters_path",
        default="esm2_t36_3B_UR50D",
        help="Path for the ESM parameters or the name of the ESM model of interest.",
    )
    parser.add_argument(
        "-bk",
        "--backbone_atoms",
        help="Use only backbone atoms for pocket definition. Default: True",
        default=True,
        type=str2bool,
    )
    parser.add_argument(
        "-r",
        "--pocket_distance_cutoff",
        help="Give a cutoff for pocket definition for filtering after Voronoi tessellation. Default: 8A",
        default=8.0,
    )
    parser.add_argument(
        "-np",
        "--number_of_processors",
        help="Numer of processors to run in parallel. Default=1",
        default=1,
    )
    parser.add_argument(
        "-se",
        "--skip_esm_run",
        help="Skip running ESM on the pocket sequences and tries to generate "
        "pocket embeddings from the existing protein embeddings. Default: False",
        default=False,
    )
    parser.add_argument(
        "-sr",
        "--select_representatives",
        help="Select representative pockets for each protein. Default: True",
        default=True,
        type=str2bool,
    )
    parser.add_argument(
        "-dm",
        "--save_distance_matrix",
        help="Save distance matrix. Default: False",
        default=False,
    )
    parser.add_argument(
        "-lm",
        "--linkage_method",
        help="Linkage method for clustering. Default: single",
        default="single",
    )
    parser.add_argument(
        "-cmin",
        "--min_threshold_for_cluster_distance",
        help="Minimum distance threshold for clusters. Default: 1.",
        default=1.0,
    )
    parser.add_argument(
        "-cmax",
        "--max_threshold_for_cluster_distance",
        help="Maximum distance threshold for clusters. Default: 3.",
        default=3.0,
    )
    parser.add_argument(
        "-cinc",
        "--increment_for_cluster_distance",
        help="Increment amount for distance threshold for clusters. Default: 0.2",
        default=0.2,
    )
    parser.add_argument(
        "-pe",
        "--path_for_embeddings",
        help="Path for saving of embeddings for the whole chain. Default: ./embeddings",
        default=f"{os.getcwd()}/embeddings/",
    )
    parser.add_argument(
        "-ppe",
        "--path_for_pocket_embeddings",
        help="Path for saving of embeddings for the pocket. Default: ./pocket_embeddings",
        default=f"{os.getcwd()}/pocket_embeddings/",
    )
    parser.add_argument(
        "-ppr",
        "--path_for_pocket_residues",
        help="Path for saving a list of pocket residues. Default: ./pocket_residues/",
        default=f"{os.getcwd()}/pocket_residues/",
    )
    parser.add_argument(
        "-ptf",
        "--path_for_tmp_folder",
        help="Path for temporary folder. Default: ./tmp/",
        default=f"{os.getcwd()}/tmp/",
    )
    parser.add_argument(
        "-ps",
        "--path_for_sequences",
        help="Path for extracted sequences. Default: ./sequences/",
        default=f"{os.getcwd()}/sequences/",
    )
    parser.add_argument(
        "-pc",
        "--path_for_clusters",
        help="Path for saving clusters. Default: ./clusters/",
        default=f"{os.getcwd()}/clusters/",
    )
    parser.add_argument(
        "-pd",
        "--path_for_distance_matrix",
        help="Path for saving distance matrix. Default: ./",
        default=f"{os.getcwd()}",
    )
    parser.add_argument(
        "-gpu",
        "--use_gpu",
        help="Use GPU for ESM generation. Default: True",
        default=True,
        type=str2bool,
    )
    parser.add_argument(
        "-debug",
        "--debugging_mode",
        help="Keep some files helpful to debug. Default: False",
        default=False,
        type=str2bool,
    )
    return parser


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def run(args):
    pocket_list_file_path = args.file
    with open(pocket_list_file_path) as f:
        pocket_list_file = f.readlines()

    pocket_list_dic = {}
    for aline in pocket_list_file:
        protein = aline.strip().split()[0]
        ligand = aline.strip().split()[1]
        if protein not in pocket_list_dic:
            pocket_list_dic[protein] = [ligand]
        else:
            pocket_list_dic[protein].append(ligand)
    print("Number of proteins:", len(pocket_list_dic))

    # check if the given paths exist, generate otherwise.
    for path in [
        args.path_for_sequences,
        args.path_for_embeddings,
        args.path_for_pocket_embeddings,
        args.path_for_pocket_residues,
        args.path_for_clusters,
        args.path_for_tmp_folder,
    ]:
        os.mkdir(path)

    # call GetPockets class, for pocket embedding generation
    get_pockets = GetPockets(
        esm_parameters_path=args.esm_parameters_path,
        only_backbone_atoms=bool(args.backbone_atoms),
        pocket_distance_cutoff=float(args.pocket_distance_cutoff),
        tmp_folder=args.path_for_tmp_folder,
        sequences_path=args.path_for_sequences,
        embeddings_path=args.path_for_embeddings,
        pocket_embeddings_path=args.path_for_pocket_embeddings,
        pocket_residues_path=args.path_for_pocket_residues,
        use_gpu=bool(args.use_gpu),
        debugging_mode=bool(args.debugging_mode),
    )

    # extract sequences from structure files and run esm model on extracted sequences
    if not bool(args.skip_esm_run):
        get_pockets.run_sequence_extraction_in_parallel(
            pocket_list_dic, num_processes=int(args.number_of_processors)
        )
        get_pockets.run_esm()
    else:
        print(
            "Skipping ESM embedding generation part and starting from pocket"
            " embedding generation, assuming the protein embeddings are present in the embeddings folder."
        )

    # generate pocket representations
    get_pockets.run_pocket_extraction_in_parallel(
        pocket_list_dic, num_processes=int(args.number_of_processors)
    )

    # cluster pockets from the pocket embeddings
    cluster_pockets = ClusterPockets(
        linkage_method=args.linkage_method,
        pocket_embeddings_path=args.path_for_pocket_embeddings,
        save_distance_matrix=bool(args.save_distance_matrix),
        min_cluster_dist=float(args.min_threshold_for_cluster_distance),
        max_cluster_dist=float(args.max_threshold_for_cluster_distance),
        increment=float(args.increment_for_cluster_distance),
        clusters_path=args.path_for_clusters,
        distance_matrix_path=args.path_for_distance_matrix,
        tmp_folder=args.path_for_tmp_folder,
        representative_selection=args.select_representatives,
        num_processes=int(args.number_of_processors),
        debugging_mode=bool(args.debugging_mode),
    )
    cluster_pockets.get_clusters()


if __name__ == "__main__":
    parser = make_argument_parser()
    args = parser.parse_args()
    run(args)
