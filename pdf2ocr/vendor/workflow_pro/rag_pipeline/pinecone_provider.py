# Filepath: code_migration/integration/rag_pipeline/pinecone_provider.py
# Description: Pinecone Provider - Vector Database Operations
# Layer: Integration
# References: reference_codebase/RAG/provider/pinecone_provider.py

"""
Pinecone Provider - Vector Database Operations

Provides access to Pinecone vector database:
- Index management (create, delete, configure)
- Vector operations (upsert, query, fetch, delete)
- Namespace management
- Metadata filtering and hybrid search
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple, Callable
from .base import BaseProvider, ProviderFactory


class PineconeProvider(BaseProvider):
    """
    Pinecone Vector Database Provider.

    Features:
    - Serverless and pod-based indexes
    - Vector CRUD operations
    - Namespace management
    - Metadata filtering
    - Hybrid search (semantic + keyword)

    Compliant with Pinecone SDK v7.x / API 2025-04.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        source_tag: Optional[str] = None
    ):
        """
        Initialize Pinecone provider.

        Args:
            api_key: Pinecone API key (defaults to PINECONE_API_KEY env var)
            source_tag: Optional source tag for tracking
        """
        try:
            from pinecone import Pinecone, ServerlessSpec
            self._Pinecone = Pinecone
            self._ServerlessSpec = ServerlessSpec
        except ImportError:
            raise ImportError("Pinecone SDK not installed. Run: pip install 'pinecone>=7.0.0'")

        self.api_key = api_key or os.getenv('PINECONE_API_KEY')
        if not self.api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY env var.")

        init_kwargs = {'api_key': self.api_key}
        if source_tag:
            init_kwargs['source_tag'] = source_tag

        self.pc = self._Pinecone(**init_kwargs)
        self.index = None
        self.index_name = None
        self.index_host = None

    def health_check(self) -> bool:
        """Check if Pinecone API is accessible."""
        try:
            self.pc.list_indexes()
            return True
        except Exception:
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get provider information."""
        return {
            'provider': 'pinecone',
            'current_index': self.index_name,
            'capabilities': ['vector_search', 'metadata_filtering', 'namespaces'],
            'indexes': self.list_indexes()
        }

    # =========================================================================
    # INDEX MANAGEMENT
    # =========================================================================

    def list_indexes(self) -> List[Dict[str, Any]]:
        """List all indexes."""
        indexes = self.pc.list_indexes()
        return [
            {
                'name': idx.name,
                'dimension': idx.dimension,
                'metric': idx.metric,
                'host': idx.host,
                'status': idx.status.state if hasattr(idx.status, 'state') else 'Unknown'
            }
            for idx in indexes
        ]

    def create_index(
        self,
        name: str,
        dimension: int = 1536,
        metric: str = 'cosine',
        cloud: str = 'aws',
        region: str = 'us-east-1',
        deletion_protection: str = 'disabled',
        tags: Optional[Dict[str, str]] = None,
        delete_if_exists: bool = False
    ) -> Any:
        """
        Create a new serverless index.

        Args:
            name: Index name
            dimension: Vector dimension (1536 for text-embedding-3-small)
            metric: Distance metric ('cosine', 'euclidean', 'dotproduct')
            cloud: Cloud provider ('aws', 'gcp', 'azure')
            region: Cloud region
            deletion_protection: 'enabled' or 'disabled'
            tags: Optional tags
            delete_if_exists: Delete existing index first

        Returns:
            Index object
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if name in existing_indexes:
            if delete_if_exists:
                self.pc.delete_index(name)
                time.sleep(1)
            else:
                return self.connect(name)

        create_kwargs = {
            'name': name,
            'dimension': dimension,
            'metric': metric,
            'spec': self._ServerlessSpec(cloud=cloud, region=region),
            'deletion_protection': deletion_protection
        }
        if tags:
            create_kwargs['tags'] = tags

        self.pc.create_index(**create_kwargs)

        # Wait for ready
        while True:
            desc = self.pc.describe_index(name)
            if desc.status.ready:
                break
            time.sleep(2)

        return self.connect(name)

    def connect(self, index_name: str) -> Any:
        """
        Connect to an existing index.

        Args:
            index_name: Name of index to connect to

        Returns:
            Index object
        """
        self.index_name = index_name
        desc = self.pc.describe_index(index_name)
        self.index_host = desc.host
        self.index = self.pc.Index(host=self.index_host)
        return self.index

    def delete_index(self, name: str) -> bool:
        """Delete an index."""
        try:
            self.pc.delete_index(name)
            if self.index_name == name:
                self.index = None
                self.index_name = None
            return True
        except Exception:
            return False

    def describe_index(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get index description."""
        idx_name = name or self.index_name
        if not idx_name:
            raise ValueError("No index name provided")

        desc = self.pc.describe_index(idx_name)
        return {
            'name': desc.name,
            'dimension': desc.dimension,
            'metric': desc.metric,
            'host': desc.host,
            'status': desc.status.state if hasattr(desc.status, 'state') else 'Unknown',
            'ready': desc.status.ready if hasattr(desc.status, 'ready') else False
        }

    def get_stats(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        """Get index statistics."""
        self._ensure_index()
        stats = self.index.describe_index_stats()
        result = {
            'total_vector_count': stats.total_vector_count,
            'dimension': stats.dimension,
            'namespaces': {}
        }
        if stats.namespaces:
            for ns, ns_stats in stats.namespaces.items():
                result['namespaces'][ns] = {'vector_count': ns_stats.vector_count}
        return result

    # =========================================================================
    # VECTOR OPERATIONS
    # =========================================================================

    def upsert(
        self,
        vectors: List[Dict[str, Any]],
        namespace: str = ""
    ) -> Dict[str, Any]:
        """
        Upsert vectors to the index.

        Args:
            vectors: List of {'id': str, 'values': List[float], 'metadata': dict}
            namespace: Target namespace

        Returns:
            Upsert result
        """
        self._ensure_index()
        result = self.index.upsert(vectors=vectors, namespace=namespace)
        return {'upserted_count': result.upserted_count}

    def query(
        self,
        vector: List[float],
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        include_values: bool = False,
        include_metadata: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query vectors by similarity.

        Args:
            vector: Query vector
            top_k: Number of results
            namespace: Namespace to search
            filter: Metadata filter
            include_values: Include vector values
            include_metadata: Include metadata

        Returns:
            List of matches
        """
        self._ensure_index()

        query_kwargs = {
            'vector': vector,
            'top_k': top_k,
            'namespace': namespace,
            'include_values': include_values,
            'include_metadata': include_metadata
        }
        if filter:
            query_kwargs['filter'] = filter

        result = self.index.query(**query_kwargs)

        return [
            {
                'id': m.id,
                'score': m.score,
                'values': m.values if include_values else None,
                'metadata': m.metadata if include_metadata else None
            }
            for m in result.matches
        ]

    def fetch(
        self,
        ids: List[str],
        namespace: str = ""
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch vectors by ID.

        Args:
            ids: Vector IDs
            namespace: Namespace

        Returns:
            Dict of id -> vector data
        """
        self._ensure_index()
        result = self.index.fetch(ids=ids, namespace=namespace)
        return {
            vid: {
                'id': v.id,
                'values': v.values,
                'metadata': v.metadata
            }
            for vid, v in result.vectors.items()
        }

    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None,
        delete_all: bool = False
    ) -> bool:
        """
        Delete vectors.

        Args:
            ids: Vector IDs to delete
            namespace: Namespace
            filter: Metadata filter for deletion
            delete_all: Delete all vectors in namespace

        Returns:
            Success status
        """
        self._ensure_index()
        try:
            if delete_all:
                self.index.delete(delete_all=True, namespace=namespace)
            elif filter:
                self.index.delete(filter=filter, namespace=namespace)
            elif ids:
                self.index.delete(ids=ids, namespace=namespace)
            return True
        except Exception:
            return False

    def update(
        self,
        id: str,
        values: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        namespace: str = ""
    ) -> bool:
        """
        Update a vector.

        Args:
            id: Vector ID
            values: New vector values
            metadata: New metadata (merged with existing)
            namespace: Namespace

        Returns:
            Success status
        """
        self._ensure_index()
        try:
            update_kwargs = {'id': id, 'namespace': namespace}
            if values:
                update_kwargs['values'] = values
            if metadata:
                update_kwargs['set_metadata'] = metadata
            self.index.update(**update_kwargs)
            return True
        except Exception:
            return False

    # =========================================================================
    # NAMESPACE OPERATIONS
    # =========================================================================

    def list_namespaces(self) -> List[str]:
        """List all namespaces in current index."""
        self._ensure_index()
        stats = self.index.describe_index_stats()
        return list(stats.namespaces.keys()) if stats.namespaces else []

    def delete_namespace(self, namespace: str) -> bool:
        """Delete all vectors in a namespace."""
        return self.delete(delete_all=True, namespace=namespace)

    # =========================================================================
    # SEARCH HELPERS
    # =========================================================================

    def search_by_text(
        self,
        text: str,
        embedding_provider: Any,
        top_k: int = 10,
        namespace: str = "",
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using text (requires embedding provider).

        Args:
            text: Search query text
            embedding_provider: Provider with embed() method
            top_k: Number of results
            namespace: Namespace to search
            filter: Metadata filter

        Returns:
            List of matches
        """
        vector = embedding_provider.embed(text)
        return self.query(vector, top_k, namespace, filter)

    def search_hybrid(
        self,
        vector: List[float],
        keywords: List[str],
        keyword_field: str = "keywords",
        top_k: int = 10,
        namespace: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: semantic + keyword filtering.

        Args:
            vector: Query vector
            keywords: Keywords to match
            keyword_field: Metadata field containing keywords
            top_k: Number of results
            namespace: Namespace

        Returns:
            Filtered matches
        """
        filter = {keyword_field: {"$in": keywords}}
        return self.query(vector, top_k, namespace, filter)

    def upsert_batch(
        self,
        vectors: List[Tuple[str, List[float], Dict[str, Any]]],
        namespace: str = "",
        batch_size: int = 100,
        on_progress: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, int]:
        """
        Upsert vectors in batches.

        Args:
            vectors: List of (id, values, metadata) tuples
            namespace: Namespace
            batch_size: Vectors per batch
            on_progress: Progress callback (current, total)

        Returns:
            Upsert statistics
        """
        self._ensure_index()
        total_upserted = 0
        total = len(vectors)

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            formatted = [
                {'id': vid, 'values': vals, 'metadata': meta}
                for vid, vals, meta in batch
            ]
            self.index.upsert(vectors=formatted, namespace=namespace)
            total_upserted += len(batch)

            if on_progress:
                on_progress(total_upserted, total)

        return {'total_upserted': total_upserted, 'namespace': namespace}

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _ensure_index(self):
        """Ensure an index is connected."""
        if self.index is None:
            raise ValueError("No index connected. Call connect() first.")


# Register with factory
ProviderFactory.register('pinecone', PineconeProvider)
