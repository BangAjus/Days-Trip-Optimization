from sanic import Sanic
from sanic.response import json

from utils import model
from utils.processing import *

app = Sanic('ML_API')

@app.route("/", methods=["GET"])
async def index(request):

    if not request:
        return json({"status":
                        { "code": 400,
                          "message":"Couldn't fetch the API"
                               }
                    },
                    status=400)
    data = request.json
    features = process_data_after_json(data)
    features_scaled = min_max_scaling(features)

    labels = model.clustering(features_scaled,
                              int(data['days']))
    labels = labels.reshape(-1, 1)
    
    features = inverse_scaling(features)
    features = concatenate((features, labels),
                           axis=1)
    features = process_to_dict(features)

    return json(features, 200)