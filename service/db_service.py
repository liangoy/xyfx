import pymongo
import config

mongodb_service=pymongo.MongoClient(config.MONGO_STR)['credit-mobile']