from motor.motor_asyncio import AsyncIOMotorClient
from ..core.config import settings
import certifi
import logging
import dns.resolver
from dotenv import load_dotenv
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

# Load environment variables from .env file

logger = logging.getLogger(__name__)

# Configure DNS resolver for MongoDB Atlas
dns.resolver.default_resolver = dns.resolver.Resolver(configure=False)
dns.resolver.default_resolver.nameservers = ['8.8.8.8']  # Use Google's DNS

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

    async def connect_to_mongodb(self):
        try:
            self.client = AsyncIOMotorClient(
                settings.MONGO_URI,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=30000,  # Increased timeout
                connectTimeoutMS=30000,
                socketTimeoutMS=30000
            )
            # Test the connection
            await self.client.admin.command('ping')
            self.db = self.client[settings.DATABASE_NAME]
            logger.info("Successfully connected to MongoDB")
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise

    async def close_mongodb_connection(self):
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")

    def get_collection(self, collection_name: str):
        if not self.db:
            raise ConnectionError("MongoDB connection not established")
        return self.db[collection_name]

mongodb = MongoDB() 