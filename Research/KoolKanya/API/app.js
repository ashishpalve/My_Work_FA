const Express = require("express");
const BodyParser = require("body-parser");
const MongoClient = require("mongodb").MongoClient;
const ObjectId = require("mongodb").ObjectID;
const CONNECTION_URL = "mongodb+srv://dev-admin:WOXFDIOOYQEF2Rar@staging.wztxj.mongodb.net/prod-dump?retryWrites=true&w=majority"
const DATABASE_NAME = "prod-dump";


var app = Express();
app.use(BodyParser.json());
app.use(BodyParser.urlencoded({ extended: true }));
var database, collection, collection2, collection3;

app.listen(5000, () => {
    MongoClient.connect(CONNECTION_URL, { useNewUrlParser: true }, (error, client) => {
        if(error) {
            throw error;
        }
        database = client.db(DATABASE_NAME);
        collection1 = database.collection("user_recommendations");
        collection2 = database.collection("popularity_reco");
        collection3 = database.collection("item_recommendations");
        console.log("Connected to `" + DATABASE_NAME + "`!");
    });
});

app.get("/user_recommendations/:id", (request, response) => {
    collection1.findOne({ "userId": new ObjectId(request.params.id)}, (error, result) => {
        if(error | !result) {
            collection1.findOne({ "model": "Popularity"}, (error, result) => {
        		if(error) {
            		return response.status(500).send(error);
        		}
        		else {
        			response.send(result['contentId']);
        			console.log("Data Fetched `" + result + "`!");
        		}
        
    		});
        }
        else {
        	response.send(result['contentId']);
        	console.log("Data Fetched `" + result + "`!");
        }
        
    });
});


app.get("/item_recommendations/:id", (request, response) => {
    collection3.findOne({ "contentId": new ObjectId(request.params.id)}, (error, result) => {
        if(error) {
            return response.status(500).send(error);
        }
        else {
            response.send(result['similar_contentId']);
            console.log("Data Fetched `" + result + "`!");
        }
        
    });
});