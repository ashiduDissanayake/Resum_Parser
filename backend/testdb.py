import asyncio
from motor.motor_asyncio import AsyncIOMotorClient
import certifi

async def test_connection():
    client = AsyncIOMotorClient(
        "mongodb+srv://ashidudissanayake1:chP0CyGcYR89zDeg@cluster0.4dg71cd.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        tlsCAFile=certifi.where()
    )
    try:
        # Test the connection
        await client.admin.command('ping')
        print("Connection successful!")
    except Exception as e:
        print(f"Connection failed: {str(e)}")
    finally:
        client.close()

asyncio.run(test_connection())