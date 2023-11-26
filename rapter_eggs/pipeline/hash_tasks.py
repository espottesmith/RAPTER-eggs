import copy
from typing import Dict, Optional

from monty.json import jsanitize

from pymatgen.core.structure import Molecule
from pymatgen.analysis.graphs import MoleculeGraph
from pymatgen.analysis.local_env import OpenBabelNN, metal_edge_extender
from pymatgen.util.graph_hashing import weisfeiler_lehman_graph_hash

from maggma.stores import Store

from rapter_eggs.schema.reactions import METALS


def hash_tasks(
    collection: Store,
    query: Optional[Dict] = None,
    batch_size: int = 1000
):
    """
    Utility function to add hashes to Jaguar calculations that lack them.

    :param collection (Store): store containing Jaguar tasks
    :param query (Optional[Dict]): query to select Jaguar tasks
    :param batch_size (int): Number of tasks to process at a time before writing to "collection".
        Useful for reducing memory burden. Default is 1000.

    :return: None
    """

    query = query if query else dict()

    updated_docs = list()
    for i, d in enumerate(collection.query(query)):
        if d["job_type"] in ["opt", "ts"]:
            mol = d.get("output", dict()).get("molecule")
        else:
            mol = d.get("input", dict()).get("molecule")

        if mol is None:
            continue

        mol = Molecule.from_dict(mol)

        mg = metal_edge_extender(MoleculeGraph.with_local_env_strategy(mol, OpenBabelNN()))

        metals = [i for i, e in enumerate(mol.species) if str(e) in METALS]

        to_delete = list()
        for bond in mg.graph.edges():
            if bond[0] in metals or bond[1] in metals:
                to_delete.append((bond[0], bond[1]))

        mg_nometal = copy.deepcopy(mg)
        for b in to_delete:
            mg_nometal.break_edge(b[0], b[1], allow_reverse=True)

        mg_undir = mg.graph.to_undirected()
        mg_nometal_undir = mg_nometal.graph.to_undirected()

        coord_hash = weisfeiler_lehman_graph_hash(
            mg_undir,
            node_attr="coords"
        )
        species_hash = weisfeiler_lehman_graph_hash(
            mg_undir,
            node_attr="specie"
        )

        hash_nometal = weisfeiler_lehman_graph_hash(
            mg_nometal_undir,
            node_attr="specie"
        )

        del d["_id"]

        d["coord_hash"] = coord_hash
        d["species_hash"] = species_hash
        d["species_hash_nometal"] = hash_nometal

        updated_docs.append(d)

        if i > 0 and i % batch_size == 0:
            print(f"{i} docs finished (updated_docs length: {len(updated_docs)}). Writing now...")
            updated_docs = jsanitize(updated_docs, allow_bson=True)
            collection.remove_docs({"calcid": {"$in": [x["calcid"] for x in updated_docs]}})
            collection.update(docs=updated_docs, key=["calcid"])
            updated_docs = list()

    if len(updated_docs) > 0:
        collection.remove_docs({"calcid": {"$in": [x["calcid"] for x in updated_docs]}})
        collection.update(docs=updated_docs, key=["calcid"])        
