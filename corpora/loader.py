"""
Corpus Loader - Load concept clusters for geometry experiments.

Loads concept clusters WITHOUT making assumptions about which geometries
fit which concepts. Experiments discover that.
"""

import yaml
from pathlib import Path
from typing import List, Optional, Set
from dataclasses import dataclass


@dataclass
class ConceptCluster:
    """A cluster of semantically related expressions."""
    domain: str
    subdomain: str
    concept: str
    expressions: List[str]
    
    def __len__(self):
        return len(self.expressions)
    
    def __repr__(self):
        return f"ConceptCluster({self.domain}/{self.subdomain}: {self.concept}, n={len(self)})"


@dataclass
class ConceptCorpus:
    """Collection of concept clusters."""
    corpus_id: str
    language: str
    version: str
    clusters: List[ConceptCluster]
    
    def __len__(self):
        return len(self.clusters)
    
    def filter_by_domain(self, domain: str) -> List[ConceptCluster]:
        """Get all clusters in a specific domain."""
        return [c for c in self.clusters if c.domain == domain]
    
    def filter_by_subdomain(self, subdomain: str) -> List[ConceptCluster]:
        """Get all clusters in a specific subdomain."""
        return [c for c in self.clusters if c.subdomain == subdomain]
    
    def get_all_expressions(self) -> List[str]:
        """Flatten all expressions across all clusters."""
        return [expr for cluster in self.clusters for expr in cluster.expressions]
    
    @property
    def domains(self) -> Set[str]:
        """Get all unique domains."""
        return {c.domain for c in self.clusters}
    
    @property
    def subdomains(self) -> Set[str]:
        """Get all unique subdomains."""
        return {c.subdomain for c in self.clusters}
    
    def __repr__(self):
        return f"ConceptCorpus({self.corpus_id}, n_clusters={len(self)}, n_expressions={len(self.get_all_expressions())})"


def load_corpus(corpus_name: str, corpora_dir: Optional[Path] = None) -> ConceptCorpus:
    """
    Load a concept cluster corpus from YAML.
    
    Args:
        corpus_name: Name of corpus file (without .yml extension)
        corpora_dir: Directory containing corpus files (defaults to ./corpora/)
    
    Returns:
        ConceptCorpus with all clusters loaded
    
    Example:
        >>> corpus = load_corpus("semantic_concepts_v0")
        >>> print(corpus.domains)
        {'emotion', 'perception', 'space', 'time', ...}
        >>> emotion_clusters = corpus.filter_by_domain("emotion")
    """
    if corpora_dir is None:
        corpora_dir = Path(__file__).parent
    
    corpus_path = corpora_dir / f"{corpus_name}.yml"
    
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    clusters = []
    for entry in data.get('entries', []):
        cluster = ConceptCluster(
            domain=entry['domain'],
            subdomain=entry['subdomain'],
            concept=entry['concept'],
            expressions=entry['expressions']
        )
        clusters.append(cluster)
    
    return ConceptCorpus(
        corpus_id=data['corpus_id'],
        language=data['language'],
        version=str(data['version']),
        clusters=clusters
    )


if __name__ == "__main__":
    import sys
    
    # Load corpus
    try:
        corpus = load_corpus("semantic_concepts_v0")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Basic stats
    print(f"Corpus: {corpus.corpus_id} (v{corpus.version})")
    print(f"Language: {corpus.language}")
    print(f"Clusters: {len(corpus)}")
    print(f"Total expressions: {len(corpus.get_all_expressions())}")
    print()
    
    # Domain breakdown
    print("Domains:")
    for domain in sorted(corpus.domains):
        clusters = corpus.filter_by_domain(domain)
        total_expr = sum(len(c) for c in clusters)
        print(f"  {domain}: {len(clusters)} clusters, {total_expr} expressions")
    print()
    
    # Sample cluster
    print("Sample cluster:")
    first = corpus.clusters[0]
    print(f"  Domain: {first.domain}")
    print(f"  Subdomain: {first.subdomain}")
    print(f"  Concept: {first.concept}")
    print(f"  Size: {len(first)} expressions")
    print(f"  Examples: {', '.join(first.expressions[:8])}")
    if len(first) > 8:
        print(f"  ... and {len(first) - 8} more")