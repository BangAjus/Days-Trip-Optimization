from numpy import append, concatenate, array
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def process_data_after_json(data):

    coordinates = data['data']
    lat, long = array([]), array([])

    for data in coordinates:

        lat = append(lat, data['latitude'])
        long = append(long, data['longitude'])

    lat = lat.reshape(-1, 1)
    long = long.reshape(-1, 1)
    coordinates = concatenate((lat, long),
                                 axis=1)

    return coordinates

def min_max_scaling(data):

    return scaler.fit_transform(data)

def inverse_scaling(data):
    
    return scaler.inverse_transform(data)

def process_to_dict(data):

    if data.shape[1] == 2:
        return {'data':[{'latitude':i, 'longitude':j} \
                        for i, j in zip(data[:, 0],
                                        data[:, 1])]}

    elif data.shape[1] == 3:
        return {'data':[{'latitude':i,
                         'longitude':j,
                         'label':k} \
                        for i, j, k in \
                            zip(data[:, 0],
                                data[:, 1],
                                data[:, 2])]}