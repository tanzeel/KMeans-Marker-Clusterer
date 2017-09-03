from django.shortcuts import render
from django.http import HttpResponse
from mcluster.models import Sale
from django.core.serializers import serialize
import pandas as pd
import json
from datetime import datetime
import calendar
from sklearn.cluster import KMeans
import time

# Create your views here.
def index(request):

    context = {}

    return render(request, 'clust/index.html', context)

def return_response_markers(request):
    t0 = time.time()

    top = request.GET['top']
    bottom = request.GET['bottom']
    right = request.GET['right']
    left = request.GET['left']

    database_list = Sale.objects.\
        filter(latitude__gte=bottom).\
        filter(latitude__lte=top). \
        filter(longitude__gte=left).\
        filter(longitude__lte=right)

    if int(request.GET['zoom']) > 13 or len(database_list) < 2000:

        b = calendar.timegm(datetime.strptime('2010-01-01', '%Y-%m-%d').timetuple())

        result = serialize('json', database_list,
                           fields=('uid', 'latitude', 'longitude', 'price', 'sale_date'))

        res = json.loads(result)
        for i in range(len(res)):
            t = calendar.timegm(datetime.strptime(res[i]['fields']['sale_date'][:10], '%Y-%m-%d').timetuple())
            res[i]['fields']['sale_date'] = str(t - b)

        return HttpResponse(json.dumps(res))

    cluster_data = {'lat': [], 'lng': []}

    for point in database_list:
        cluster_data['lat'].append(point.latitude)
        cluster_data['lng'].append(point.longitude)

    df = pd.DataFrame(cluster_data, columns=['lat', 'lng'], index=None)

    kmeans = KMeans(n_clusters=40, n_init=1, max_iter=30, tol=0.001, n_jobs=2).fit(df)
    # kmeans = KMeans(n_clusters=40).fit(df)
    samples = kmeans.predict(df)

    centroids = kmeans.cluster_centers_

    response_data = []
    for i in range(len(centroids)):
        response_data.append([centroids[i][0], centroids[i][1], len([j for j in samples if j == i])])


    response = {'data': response_data, 'cluster': 'true'}

    print(time.time() - t0)
    print(len(database_list))

    return HttpResponse(json.dumps(response))