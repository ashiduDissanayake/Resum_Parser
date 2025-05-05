from motor.motor_asyncio import AsyncIOMotorClient
from ..core.config import settings

class MongoDB:
    client: AsyncIOMotorClient = None
    db = None

    async def connect_to_mongodb(self):
        self.client = AsyncIOMotorClient(settings.MONGO_URI)
        self.db = self.client[settings.DATABASE_NAME]

    async def close_mongodb_connection(self):
        if self.client:
            self.client.close()

    def get_collection(self, collection_name: str):
        return self.db[collection_name]

mongodb = MongoDB() 