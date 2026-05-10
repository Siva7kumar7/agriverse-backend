import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# SINGLE DATABASE CONNECTION — ALL modules import from here
# =====================================================
mongo_uri = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
client = MongoClient(mongo_uri)
db = client["agriverse_db"]

# Collections
user_collection = db["users"]
product_collection = db["products"]
cart_collection = db["cart"]
order_collection = db["orders"]
