"""
Module for handling pre-orders and partial orders.
"""
from typing import Set, Tuple, List, Any, Iterator
from itertools import combinations, permutations

import networkx as nx


class PreOrder:
    """
    Represents a pre-order (reflexive and transitive relation).
    Stored as a dictionary of comparable pairs.
    """
    elements: Set[Any]
    relations: Set[Tuple[Any, Any]] # (a, b) means a <= b
    hasse_diagram: nx.DiGraph # One node per equivalence class, each equivalence class is a set of elements

    def __init__(self, elements: Set[Any], relations: Set[Tuple[Any, Any]]):
        relations = _transitive_closure(elements, relations=relations)
        relations = _reflexive_closure(elements, relations=relations)

        self.elements = elements
        self.relations = relations  # (a, b) means a <= b
        self._build_hasse_diagram()

        # Validate may use the hasse diagram or other properties, always validate last
        self._validate()
    
    def build_sub_preorder(self, subset: Set[Any]) -> 'PreOrder':
        """
        Build the sub-preorder induced on the given subset of elements.
        """
        assert subset.issubset(self.elements), "Subset must be within the elements of the preorder"
        sub_relations = set()
        for a, b in self.relations:
            if a in subset and b in subset:
                sub_relations.add((a, b))
        return PreOrder(subset, sub_relations)

    def leq(self, a: Any, b: Any) -> bool:
        """Check if a <= b"""
        return (a, b) in self.relations

    def less(self, a: Any, b: Any) -> bool:
        """Check if a < b (strict)"""
        return self.leq(a, b) and not self.leq(b, a)

    def geq(self, a: Any, b: Any) -> bool:
        """Check if a >= b"""
        return self.leq(b, a)

    def greater(self, a: Any, b: Any) -> bool:
        """Check if a > b (strict)"""
        return self.less(b, a)

    def _validate(self):
        # Check reflexivity
        for elem in self.elements:
            if (elem, elem) not in self.relations:
                raise ValueError(f"Pre-order not reflexive: missing ({elem}, {elem})")

        # Check transitivity
        for a, b, c in combinations(self.elements, 3):
            if (a, b) in self.relations and (b, c) in self.relations:
                if (a, c) not in self.relations:
                    raise ValueError(f"Pre-order not transitive: missing ({a}, {c}) from ({a}, {b}) and ({b}, {c})")
    
    def _build_hasse_diagram(self):
        """Build the Hasse diagram for the pre-order."""
        G = nx.DiGraph()
        # compute the equivalence classes, add them as nodes
        # a and b are in the same equivalence class if a <= b and b <= a
        equivalence_classes: List[Set[Any]] = []
        for e in self.elements:
            found_eq_class_for_e = False
            for eq_class in equivalence_classes:
                representative = next(iter(eq_class))
                if self.leq(e, representative) and self.leq(representative, e):
                    eq_class.add(e)
                    found_eq_class_for_e = True
                    break
            if not found_eq_class_for_e:
                equivalence_classes.append(set([e]))
        # add nodes
        for eq_class in equivalence_classes:
            G.add_node(frozenset(eq_class))
        # add all leq relations between equivalence classes, later compute the transitive reduction
        for class_a, class_b in combinations(equivalence_classes, 2):
            a_rep = next(iter(class_a))
            b_rep = next(iter(class_b))
            if self.less(a_rep, b_rep):
                G.add_edge(frozenset(class_a), frozenset(class_b))
            elif self.less(b_rep, a_rep):
                G.add_edge(frozenset(class_b), frozenset(class_a))
        # compute transitive reduction to get Hasse diagram
        self.hasse_diagram = nx.transitive_reduction(G)

    def __hash__(self):
        return hash(tuple([frozenset(self.elements), frozenset(self.relations)]))
    
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, PreOrder):
            return False
        return self.elements == value.elements and self.relations == value.relations

    def __repr__(self) -> str:
        # Use edges of the Hasse diagram for concise representation
        edges = list(self.hasse_diagram.edges())
        edges_to_show = []
        # if node has only one element, show that element instead of frozenset
        # if node has multiple elements, show as a set with curly braces
        for edge in edges:
            u, v = edge
            if len(u) == 1:
                u = next(iter(u))
            else:
                u = "{" + ", ".join(map(str, u)) + "}"
            if len(v) == 1:
                v = next(iter(v))
            else:
                v = "{" + ", ".join(map(str, v)) + "}"
            edges_to_show.append((u, v))
        return f"PreOrder(Elements: {self.elements}, Hasse edges: {edges_to_show})"

class PartialOrder(PreOrder):
    """
    Represents a partial order (reflexive, transitive, antisymmetric).
    """
    def _validate(self):
        PreOrder._validate(self)
        # Check antisymmetry
        for a, b in combinations(self.elements, 2):
            if (a, b) in self.relations and (b, a) in self.relations and a != b:
                raise ValueError(f"Pre-order not antisymmetric: both ({a}, {b}) and ({b}, {a}) for a != b")
    
    def __repr__(self) -> str:
        edges = list(self.hasse_diagram.edges())
        # Here every node is a singleton set since it's a partial order
        edges_to_show = []
        for edge in edges:
            u, v = edge
            u = next(iter(u))  # singleton
            v = next(iter(v))  # singleton
            edges_to_show.append((u, v))
        return f"PartialOrder(Elements: {self.elements}, Hasse edges: {edges_to_show})"

def total_order_from_list(elements: List[Any]) -> PartialOrder:
    """
    Create a total order from a list (first element is smallest).
    """
    relations = set()
    for i, a in enumerate(elements):
        for j, b in enumerate(elements):
            if i <= j:
                relations.add((a, b))
    return PartialOrder(set(elements), relations)

def minimal_elements(poset: PreOrder) -> Set[Any]:
    """
    Find minimal elements in a pre-order/partial order.
    """
    minimals = set()
    for node in _source_nodes(poset.hasse_diagram):
        minimals.update(node)
    return minimals

def maximal_elements(poset: PreOrder) -> Set[Any]:
    """
    Find maximal elements in a pre-order/partial order.
    """
    maximals = set()
    for node in _terminal_nodes(poset.hasse_diagram):
        maximals.update(node)
    return maximals

def completions_of_poset(poset: PartialOrder) -> Iterator[PartialOrder]:
    """
    Generate all total order completions of the given partial order.
    Note: This can be computationally expensive for large posets.
    """
    # Generate all linear extensions via backtracking on minimal elements.
    # Build adjacency and indegree from the Hasse diagram (cover relations).
    node_to_elem = {}
    for node in poset.hasse_diagram.nodes():
        if isinstance(node, frozenset):
            if len(node) != 1:
                raise ValueError("PartialOrder has non-singleton node in Hasse diagram")
            elem = next(iter(node))
        else:
            elem = node
        node_to_elem[node] = elem

    adj = {node_to_elem[n]: set() for n in poset.hasse_diagram.nodes()}
    indeg = {node_to_elem[n]: 0 for n in poset.hasse_diagram.nodes()}
    for u, v in poset.hasse_diagram.edges():
        ue = node_to_elem[u]
        ve = node_to_elem[v]
        if ve not in adj[ue]:
            adj[ue].add(ve)
            indeg[ve] += 1

    def _backtrack(order: List[Any], available: Set[Any], current_indeg: dict) -> Iterator[PartialOrder]:
        if len(order) == len(adj):
            yield total_order_from_list(order)
            return
        for elem in list(available):
            new_available = set(available)
            new_available.remove(elem)
            new_indeg = dict(current_indeg)
            for succ in adj[elem]:
                new_indeg[succ] -= 1
                if new_indeg[succ] == 0:
                    new_available.add(succ)
            yield from _backtrack(order + [elem], new_available, new_indeg)

    initial_available = {e for e, d in indeg.items() if d == 0}
    yield from _backtrack([], initial_available, indeg)

def _source_nodes(diagram: nx.DiGraph) -> List[Any]:
    sources = []
    for node, degree in diagram.in_degree():
        if degree == 0:
            sources.append(node)
    return sources

def _terminal_nodes(diagram: nx.DiGraph) -> List[Any]:
    terminals = []
    for node, degree in diagram.out_degree():
        if degree == 0:
            terminals.append(node)
    return terminals

def _transitive_closure(elements: Set[Any], relations: Set[Tuple[Any, Any]]) -> Set[Tuple[Any, Any]]:
    """
    Compute the transitive closure of the given relations.
    """
    closure = set(relations)
    added = True
    while added:
        added = False
        for a, b, c in permutations(elements, 3):
            if (a, b) in closure and (b, c) in closure and (a, c) not in closure:
                closure.add((a, c))
                added = True
    return closure

def _reflexive_closure(elements: Set[Any], relations: Set[Tuple[Any, Any]]) -> Set[Tuple[Any, Any]]:
    """
    Compute the reflexive closure of the given relations.
    """
    closure = set(relations)
    for e in elements:
        closure.add((e, e))
    return closure
