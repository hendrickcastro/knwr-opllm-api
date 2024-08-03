from pymongo import MongoClient
from bson import ObjectId
from ...core.config import settings
from ...core.utils import setup_logger
from typing import Dict, Any, List
import traceback
import time

logger = setup_logger(__name__)

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.connect()

    def connect(self):
        try:
            logger.info(f"Attempting to connect to database at {settings.DATABASE_URL}")
            self.client = MongoClient(settings.DATABASE_URL, serverSelectionTimeoutMS=5000)
            self.client.server_info()  # This will raise an exception if it can't connect
            self.db = self.client[settings.MONGODB_DB]
            logger.info("Database connection established successfully")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            self.client = None
            self.db = None

    def _ensure_connection(self):
        if self.client is None or self.db is None:
            self.connect()
        if self.client is None or self.db is None:
            raise Exception("No database connection available")

    def _serialize_object_id(self, obj):
        if isinstance(obj, dict):
            return {key: self._serialize_object_id(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_object_id(item) for item in obj]
        elif isinstance(obj, ObjectId):
            return str(obj)
        return obj

    def store_embedding(self, embedding: List[float], metadata: Dict[str, Any]) -> str:
        try:
            self._ensure_connection()
            collection = self.db.embeddings
            logger.info(f"Storing embedding in collection: {collection.name}")
            result = collection.insert_one({"embedding": embedding, "metadata": metadata})
            embedding_id = str(result.inserted_id)
            logger.info(f"Stored embedding with id: {embedding_id}")
            return embedding_id
        except Exception as e:
            logger.error(f"Error storing embedding: {str(e)}")
            logger.error(traceback.format_exc())
            return None  # Return None instead of raising an exception

    def search_similar_embeddings(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            self._ensure_connection()
            collection = self.db.embeddings
            logger.info(f"Searching for similar embeddings in collection: {collection.name}")
            logger.info(f"Query embedding length: {len(query_embedding)}")
            
            pipeline = [
                {
                    "$addFields": {
                        "dot_product": {
                            "$reduce": {
                                "input": {"$zip": {"inputs": ["$embedding", query_embedding]}},
                                "initialValue": 0,
                                "in": {"$add": ["$$value", {"$multiply": [{"$arrayElemAt": ["$$this", 0]}, {"$arrayElemAt": ["$$this", 1]}]}]}
                            }
                        },
                        "magnitude_a": {
                            "$sqrt": {
                                "$reduce": {
                                    "input": "$embedding",
                                    "initialValue": 0,
                                    "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                }
                            }
                        },
                        "magnitude_b": {
                            "$sqrt": {
                                "$reduce": {
                                    "input": query_embedding,
                                    "initialValue": 0,
                                    "in": {"$add": ["$$value", {"$multiply": ["$$this", "$$this"]}]}
                                }
                            }
                        }
                    }
                },
                {
                    "$addFields": {
                        "cosine_similarity": {
                            "$divide": ["$dot_product", {"$multiply": ["$magnitude_a", "$magnitude_b"]}]
                        }
                    }
                },
                {"$sort": {"cosine_similarity": -1}},
                {"$limit": top_k},
                {
                    "$project": {
                        "_id": 1,
                        "metadata": 1,
                        "cosine_similarity": 1
                    }
                }
            ]
            
            logger.info("Executing aggregation pipeline...")
            results = list(collection.aggregate(pipeline))
            logger.info(f"Found {len(results)} similar embeddings")
            
            # Ensure the correct format of the results and handle potential missing fields
            formatted_results = []
            for result in results:
                formatted_result = {
                    "_id": str(result["_id"]),
                    "metadata": result.get("metadata", {}),
                    "cosine_similarity": result.get("cosine_similarity", 0.0)
                }
                formatted_results.append(formatted_result)
                logger.info(f"Formatted result: {formatted_result}")
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching similar embeddings: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def store_document(self, content: str, metadata: Dict[str, Any]) -> str:
        try:
            self._ensure_connection()
            collection = self.db.documents
            logger.info(f"Storing document in collection: {collection.name}")
            result = collection.insert_one({"content": content, "metadata": metadata})
            document_id = str(result.inserted_id)
            logger.info(f"Stored document with id: {document_id}")
            return document_id
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def get_document_by_id(self, document_id: str) -> Dict[str, Any]:
        try:
            self._ensure_connection()
            collection = self.db.documents
            document = collection.find_one({"_id": ObjectId(document_id)})
            if document:
                return self._serialize_object_id(document)
            else:
                logger.warning(f"Document with id {document_id} not found")
                return None
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        try:
            self._ensure_connection()
            collection = self.db.documents
            result = collection.update_one(
                {"_id": ObjectId(document_id)},
                {"$set": {"content": content, "metadata": metadata}}
            )
            if result.modified_count > 0:
                logger.info(f"Updated document with id: {document_id}")
                return True
            else:
                logger.warning(f"Document with id {document_id} not found or not modified")
                return False
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            logger.error(traceback.format_exc())
            raise

db = Database()
