import os
import numpy as np
import openai
from typing import List, Dict, Any
import logging
import asyncio
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class SemanticSearch:
    def __init__(self):
        self.openai_client = openai.AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"))
        self.index = None
        self.clauses = []
        self.embeddings = []

    async def search_clauses(self, question: str, document_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            clauses = document_data.get("clauses", [])
            if not clauses:
                return []

            question_embedding = await self._get_embedding(question)
            clause_embeddings = await self._get_clause_embeddings(clauses)

            relevant_clauses = await self._find_similar_clauses(
                question_embedding,
                clause_embeddings,
                clauses
            )

            return relevant_clauses

        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}")
            return []

    async def _get_embedding(self, text: str) -> List[float]:
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            raise Exception(f"Embedding generation failed: {str(e)}")

    async def _get_clause_embeddings(self, clauses: List[Dict[str, Any]]) -> List[List[float]]:
        embeddings = []
        for clause in clauses:
            embedding = await self._get_embedding(clause["content"])
            embeddings.append(embedding)
        return embeddings

    async def _find_similar_clauses(
        self,
        question_embedding: List[float],
        clause_embeddings: List[List[float]],
        clauses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        try:
            if not clause_embeddings:
                return []

            question_vector = np.array([question_embedding]).astype('float32')
            clause_vectors = np.array(clause_embeddings).astype('float32')

            dimension = len(question_embedding)
            index = faiss.IndexFlatIP(dimension)
            index.add(clause_vectors)

            similarities, indices = index.search(
                question_vector, min(5, len(clauses)))

            relevant_clauses = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if similarity > 0.3:
                    clause = clauses[idx].copy()
                    clause["similarity_score"] = float(similarity)
                    clause["rank"] = i + 1
                    relevant_clauses.append(clause)

            return sorted(relevant_clauses, key=lambda x: x["similarity_score"], reverse=True)

        except Exception as e:
            logger.error(f"Error finding similar clauses: {str(e)}")
            return []

    async def _batch_embeddings(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = await self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                batch_embeddings = [data.embedding for data in response.data]
                embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error in batch embedding: {str(e)}")
                for _ in batch:
                    embeddings.append([0.0] * 1536)

        return embeddings
