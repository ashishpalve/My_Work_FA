const Express = require("express");
const BodyParser = require("body-parser");
const MongoClient = require("mongodb").MongoClient;
const ObjectId = require("mongodb").ObjectID;
const CONNECTION_URL = "mongodb+srv://dev:oAX95s3vvOcqwZ4b@staging.wztxj.mongodb.net/test?authSource=admin&replicaSet=staging-shard-0&w=majority&readPreference=primary&appname=MongoDB%20Compass%20Community&retryWrites=true&ssl=true"
const DATABASE_NAME = "prod-dump";


var app = Express();
app.use(BodyParser.json());
app.use(BodyParser.urlencoded({ extended: true }));
var database, collection, collection2;

app.listen(5000, () => {
    MongoClient.connect(CONNECTION_URL, { useNewUrlParser: true }, (error, client) => {
        if(error) {
            throw error;
        }
        database = client.db(DATABASE_NAME);
        collection = database.collection("reco_test");
        collection2 = database.collection("popularity_reco");
        console.log("Connected to `" + DATABASE_NAME + "`!");
    });
});

app.get("/reco_test/:id", (request, response) => {
    collection.findOne({ "_id": new ObjectId(request.params.id)}, (error, result) => {
        if(error) {
            return response.status(500).send(error);
        }
        else {
            response.send(result['contentId']);
            console.log("Data Fetched `" + result + "`!");
        }
        
    });
});


app.get("/popularity_reco", (request, response) => {
    collection2.findOne({ "userId": "dummy"}, (error, result) => {
        if(error) {
            return response.status(500).send(error);
        }
        else {
            response.send(result['contentId']);
            console.log("Data Fetched `" + result + "`!");
        }
        
    });
});