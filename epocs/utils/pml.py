import itertools

from typing import Optional

from pymol import cmd


class UtilsPml:
    @staticmethod
    def iterate_prop(sele_str: str, prop: str = "(index, alt)") -> list:
        mspc = {"atom_info": []}
        cmd.iterate(sele_str, "atom_info.append({0})".format(prop), space=mspc)
        return mspc["atom_info"]

    @staticmethod
    def iterate_prop_set(sele_str: str, prop: str = "(index, alt)") -> set:
        mspc = {"atom_info": set([])}
        cmd.iterate(sele_str, "atom_info.add({0})".format(prop), space=mspc)
        return mspc["atom_info"]

    @staticmethod
    def run_iterate(sele_str: str, container: list, rule: str = "res.append({0})"):
        mspc = {"res": container()}
        cmd.iterate(sele_str, rule, space=mspc)
        return mspc["res"]

    @staticmethod
    def fix_alt_locs(
        sele: str,
        logging_name: str = "",
        rename_alt_locs: bool = True,
        resnames_to_check: Optional[list] = None,
    ):
        """
        Selects the alt locs with the best occupancy. Should work well only with default-named residue codes
        (i.e. any amino acid or other het-code, but probably not a custom ligand saved to PDB from mol format).
        Requires an existing object/selection in the current pymol's context to look for alt locs codes in.
        It is passed to sele.
        """
        alts = UtilsPml.iterate_prop_set(sele, "alt")
        index_alt_occ = UtilsPml.iterate_prop(
            sele, "(resi, resn, chain, segi, name, alt, q)"
        )
        if alts != {""}:
            # Creates a dictionary {(resi, resn, chain, segi): [name, alt, occ]}.
            # can be re-written as
            # for (resi, resn, chain, segi) in index_alt_occ:
            #     if (resi, resn, chain, segi) not in resi_alts_d:
            #         resi_alts_d[(resi, resn, chain, segi)]=[]... and etc
            resi_alts_d = {
                k: list([_[4:] for _ in g])
                for k, g in itertools.groupby(
                    sorted(index_alt_occ, key=lambda _: _[:4]), key=lambda _: _[:4]
                )
            }

            for (resi, resn, chain, segi), vals in resi_alts_d.items():
                if resn not in resnames_to_check:
                    continue
                if set([_[1] for _ in vals]) == {""}:
                    continue
                # Sorts by name, max occupancy, and best alt.
                # Alt '' is less than 'a', so will be on top of the sorted stuff.
                # You may change the sort order in the lambda to prioritize alphabetic id of the alt loc
                # instead of the occupancy value.
                # Atom name aka a[0] has the top sorting priority to make itertools.groupby work correctly.
                vals_sorted = sorted(vals, key=lambda a: (a[0], -float(a[2]), a[1]))
                # Creates a dictionary {name: [name, alt, occ]}.
                atom_alt_occ_sorted_d = {
                    k: list(g)
                    for k, g in itertools.groupby(
                        sorted(vals_sorted), key=lambda _: _[0]
                    )
                }

                for atom_name, atom_vals in atom_alt_occ_sorted_d.items():
                    if len(atom_vals) > 0:
                        for atom_name, alt, occ in atom_vals[1:]:
                            cmd.remove(
                                f"(resi {resi}) and (resn {resn}) and "
                                f'(chain {chain}) and (segi "{segi}") and '
                                f"(name {atom_name}) and (alt {alt})"
                            )
            if rename_alt_locs:
                cmd.alter(sele, 'alt=""')
