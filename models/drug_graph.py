class DrugSynonymGraph:
    def __init__(self):
        self._brand_to_generic: dict[str, set[str]] = {}
        self._generic_to_brands: dict[str, set[str]] = {}

    def add_brand_relation(self, brand: str, generic: str):
        self._brand_to_generic.setdefault(brand, set()).add(generic)
        self._generic_to_brands.setdefault(generic, set()).add(brand)

    def get_generic(self, brand: str) -> set[str]:
        return self._brand_to_generic.get(brand, set())

    def get_brands(self, generic: str) -> set[str]:
        return self._generic_to_brands.get(generic, set())

    def is_brand_of(self, brand: str, generic: str) -> bool:
        return generic in self._brand_to_generic.get(brand, set())

    def resolve_conflict(self, candidates: list[str]) -> str | None:
        candidate_set = set(candidates)
        for cand in candidates:
            generics = self._brand_to_generic.get(cand, set())
            if generics & candidate_set:
                return cand
        return None

    def summary(self) -> str:
        lines = []
        total = sum(len(v) for v in self._brand_to_generic.values())
        lines.append(f"Drug graph: {total} relasi brand->generic")
        for brand, generics in self._brand_to_generic.items():
            for g in generics:
                lines.append(f"  {brand} --[is_brand_of]--> {g}")
        return "\n".join(lines)


def build_drug_graph() -> DrugSynonymGraph:
    graph = DrugSynonymGraph()
    graph.add_brand_relation("Acetin",   "Acetylcysteine")
    graph.add_brand_relation("Acepress", "Acebutolol")

    return graph
