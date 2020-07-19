from flask import Flask, request, jsonify
from flask_pymongo import PyMongo

app = Flask(__name__)

app.config['MONGO_URI'] = 'mongodb+srv://dev:oAX95s3vvOcqwZ4b@staging.wztxj.mongodb.net/test?retryWrites=true&w=majority'

mongo = PyMongo(app)

print("-----------------------------------------------------------------------------------------------------")
print("\nConnecting to the database:")
from pymongo import MongoClient
# pprint library is used to make the output look more pretty
from pprint import pprint

# connect to MongoDB, change the << MONGODB URL >> to reflect your own connection string
client = MongoClient("mongodb+srv://dev:oAX95s3vvOcqwZ4b@staging.wztxj.mongodb.net/test?retryWrites=true&w=majority")
db=client['prod-dump']
collection = db.reco_test

@app.route('/api/:item_id', methods=['GET'])
def get_recommendations():
	item_id = request.values['_id']
	cursor = mongo.db.reco_test
	results = []
	for json_obj in cursor.find({'_id':item_id}):
		results.append(json_obj['contentId'])
	if(results):
		return jsonify(results)
	else:
		return jsonify('Error retrieving data')

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=9000, debug=True)
