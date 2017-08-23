from django.shortcuts import render
from django.http import HttpResponse
from django.conf import settings
from .forms import SearchForm
from prac.models import Sale, Housing, CSORef, SexAgeMarriage, ReportedErrors
import googlemaps
from django.core.serializers import serialize
import math
import numpy as np
import pandas as pd
from django.db.models import Avg, Max, Min, Count
import json
from datetime import datetime, date
import calendar
from sklearn.cluster import KMeans

def geolocate(address_string):
    """
    Take address from search bar and return lat & long location if possible.
    :param address_string:
    :return: location, status

    Shorthand for status:
    0 - No search conducted
    1 - Good search results
    2 - Bad / No results
    """
    gmaps = googlemaps.Client(key='AIzaSyBFwN-7_erzpXeWWFe3DwMqSPKGoCjj1Hg')

    # Add Ireland to search string to remove ambiguity
    address = address_string + ' Ireland'

    # Get results from gmaps API
    geocode_result = gmaps.geocode(address)

    # If a result is returned
    if len(geocode_result) > 0:
        # Set the location as the result
        location = geocode_result[0]['geometry']['location']
        location['l'] = location['lng'] - 0.04866600036624025
        location['r'] = location['lng'] + 0.04866600036624025
        location['t'] = location['lat'] + 0.019106276426853697
        location['b'] = location['lat'] - 0.019106276426853697
        status = 1
    else:
        status = 2

    # Check if location is within east/west boundary of Ireland
    if status is 1:
        if location['lng'] > -10.738539 and location['lng'] < -5.930445:
            pass
        else:
            status = 2

    # Check if location is within north/south boundary of Ireland
    if status is 1:
        if location['lat'] > 51.387652 and location['lat'] < 55.445918:
            pass
        else:
            status = 2

    # If a bad result, set default location
    if status is 2:
        location = {'lng': -6.2603, 'lat': 53.3498, 'l': -6.30896600036624,
                'r':-6.211633999633818, 't':53.368906276426856,
                'b':53.330685156427386}

    return location, status


def index(request):
    form = SearchForm()

    location = {'lng': -6.2603, 'lat': 53.3498, 'l': -6.30896600036624,
                'r':-6.211633999633818, 't':53.368906276426856,
                'b':53.330685156427386}
    status = 0

    with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'RRP_timestamp.txt', 'r') as file:
        lu = file.read()

    if request.method == 'GET':
        form = SearchForm(request.GET)
        if form.is_valid():
            search_address = form.cleaned_data['address']
            location, status = geolocate(search_address)

    database_list = Sale.objects.\
        filter(quality='good').\
        filter(latitude__gte=location['b']).\
        filter(latitude__lte=location['t']).\
        filter(longitude__gte=location['l']).\
        filter(longitude__lte=location['r'])

    datalist = Sale.objects.aggregate(Max('price'))
    max_price = datalist['price__max']

    scatter_data = {'date': [], 'price': []}
    hist_data = []

    for sale in database_list:
        raw_time = calendar.timegm(sale.sale_date.timetuple()) * 1000
        scatter_data['date'].append(sale.sale_date)
        scatter_data['price'].append(float(sale.price))
        hist_data.append(float(sale.price))

    df = pd.DataFrame(scatter_data, columns=['price'], index=scatter_data['date'])

    df.index.names = ['date']

    df = df.set_index(pd.DatetimeIndex(df.index))
    df = df.resample('W').mean().dropna(axis=0, how='any')

    scatter_data = list(map(lambda x,y: [calendar.timegm(x.to_pydatetime().timetuple()) * 1000, round(float(y),2)], df.index.tolist(), df.values))

    compressed_hist_data = {}
    k = 0
    for v in range(25000, 5000000, 25000):
        count = 0
        for i in hist_data:
            if i > k and i <= v:
                count += 1

        if count != 0:
            compressed_hist_data[v - 12500] = count

        k = v

    lobf_coef = [float(i) for i in list(np.polyfit([i[0] for i in scatter_data], [i[1] for i in scatter_data], deg=3))]

    context = {'form': form, 'latitude': location['lat'],
               'longitude':location['lng'], 'status': status,
               'database_list': database_list,
               'hist_data': compressed_hist_data,
               'scatter_data': scatter_data,
               'last_update': lu,
               'max_price': max_price, 'lobf_coef': lobf_coef}

    return render(request, 'homepage/index.html', context)


def contact(request):
    return render(request, 'homepage/contact.html')


def legal(request):
    return render(request, 'homepage/privacy-cookies.html')


def data(request):
    return render(request, 'homepage/our-data.html')


def return_response_markers(request):
    top = request.GET['top']
    bottom = request.GET['bottom']
    right = request.GET['right']
    left = request.GET['left']

    database_list = Sale.objects.\
        filter(quality='good').\
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

    kmeans = KMeans(n_clusters=40).fit(df)
    samples = kmeans.predict(df)

    centroids = kmeans.cluster_centers_

    response_data = []
    for i in range(len(centroids)):
        response_data.append([centroids[i][0], centroids[i][1], len([j for j in samples if j == i])])

    response = {'data': response_data, 'cluster': 'true'}

    return HttpResponse(json.dumps(response))


def return_response_infowindow(request):
    uid = request.GET['uid']

    database_list = Sale.objects.filter(uid=uid)

    result = serialize('json', database_list,
                       fields=('uid', 'sale_date', 'address', 'postcode', 'county',
                               'nfma', 'vat_ex', 'DoP', 'PSD', 'price'))

    return HttpResponse(result)


def latlng(clat, clng, d, dir):
    R = 6378.1
    if dir == 'N':
        brng = 0
    elif dir == 'E':
        brng = 1.57
    elif dir == 'S':
        brng = 3.14
    elif dir == 'W':
        brng = 4.71
    else:
        brng = 0

    lat1 = (clat / 180) * math.pi
    lng1 = (clng / 180) * math.pi

    lat2 = math.asin(math.sin(lat1) * math.cos(d / R) + math.cos(lat1)
                     * math.sin(d / R) * math.cos(brng))
    lng2 = lng1 + math.atan2(math.sin(brng) * math.sin(d / R) * math
                             .cos(lat1),
                             math.cos(d / R) - math.sin(lat1) * math
                             .sin(lat2))

    return [lat2 * (180 / math.pi), lng2 * (180 / math.pi)]


def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295
    a = 0.5 - math.cos((lat2 - lat1) * p) / 2 + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2

    return 12742 * math.asin(math.sqrt(a))


def retrieve_stats(queryset):
    ag_data = queryset.aggregate(Avg('price'), Min('price'), Max('price'),
                             Count('price'))

    count = ag_data['price__count']

    values = queryset.values_list('price', flat=True).order_by('price')

    if count % 2 == 1:
        med = values[int(round(count / 2 - 0.5))]
    else:
        med = sum(values[count / 2 - 0.5: count / 2 + 0.5]) / 2

    sale_list = []
    size_list = []

    scatter_data = {'date': [], 'price': []}
    hist_data = []

    for sale in queryset:
        raw_time = calendar.timegm(sale.sale_date.timetuple())
        sale_list.append(raw_time)
        scatter_data['date'].append(sale.sale_date)
        scatter_data['price'].append(float(sale.price))
        hist_data.append(float(sale.price))
        if sale.PSD == 'greater than or equal to 38 sq metres and less than 125 sq metres':
            size_list.append(81.5)
        elif sale.PSD == 'greater than 125 sq metres':
            size_list.append(125)
        elif sale.PSD == 'less than 38 sq metres':
            size_list.append(38)

    avg_sale = round(np.mean(sale_list) * 1000, 0)
    avg_size = round(np.mean(size_list), 1)

    df = pd.DataFrame(scatter_data, columns=['price'],
                      index=scatter_data['date'])
    df.index.names = ['date']

    df = df.set_index(pd.DatetimeIndex(df.index))
    df = df.resample('W').mean().dropna(axis=0, how='any')

    scatter_data = list(map(lambda x, y: [calendar.timegm(x.to_pydatetime().timetuple()) * 1000,
                      round(float(y), 2)], df.index.tolist(), df.values))

    compressed_hist_data = compress_list(hist_data, 25000)

    if max(compressed_hist_data.keys()) / 25 < 25000:
        compressed_hist_data = compress_list(hist_data, 20000)
        if max(compressed_hist_data.keys()) / 25 < 20000:
            compressed_hist_data = compress_list(hist_data, 15000)
            if max(compressed_hist_data.keys()) / 25 < 15000:
                compressed_hist_data = compress_list(hist_data, 10000)


    return {'ave_price': ag_data['price__avg'],
            'min_price': float(ag_data['price__min']),
            'max_price': float(ag_data['price__max']),
            'min_date': calendar.timegm(queryset.order_by('sale_date')[
                                         0].sale_date.timetuple()) * 1000,
            'max_date': calendar.timegm(queryset.order_by('-sale_date')[
                                         0].sale_date.timetuple()) * 1000,
            'med_price': float(med), 'avg_date': avg_sale,
            'avg_size': avg_size,  'hist_data': compressed_hist_data,
            'scatter_data': scatter_data}


def compress_list(data, limit):
    a = {}
    k = 0
    for v in range(limit, 5000000, limit):
        count = 0
        for i in data:
            if i > k and i <= v:
                count += 1

        if count != 0:
            a[v - limit/2] = count

        k = v

    b = {}
    for k, v in a.items():
        if v > max(a.values()) * 0.05:
            b[k] = v

    return b


def get_age_stats(zoom, uid, year):
    age_profile = SexAgeMarriage.objects.filter(zoom=zoom, uid=uid, year=year)[0]

    age_04 = age_profile.age_04
    age_59 = age_profile.age_59
    age_1014 = age_profile.age_1014
    age_1519 = age_profile.age_1519
    age_2024 = age_profile.age_2024
    age_2529 = age_profile.age_2529
    age_3034 = age_profile.age_3034
    age_3539 = age_profile.age_3539
    age_4044 = age_profile.age_4044
    age_4549 = age_profile.age_4549
    age_5054 = age_profile.age_5054
    age_5559 = age_profile.age_5559
    age_6064 = age_profile.age_6064
    age_6569 = age_profile.age_6569
    age_7074 = age_profile.age_7074
    age_7579 = age_profile.age_7579
    age_8084 = age_profile.age_8084
    age_85p = age_profile.age_85p

    w_age = sum([age_04 * 2.5, age_59 * 7.5, age_1014 * 12.5,
                 age_1519 * 17.5, age_2024 * 22.5, age_2529 * 27.5,
                 age_3034 * 32.5, age_3539 * 37.5, age_4044 * 42.5,
                 age_4549 * 47.5, age_5054 * 52.5, age_5559 * 57.5,
                 age_6064 * 62.5, age_6569 * 67.5, age_7074 * 72.5,
                 age_7579 * 77.5, age_8084 * 82.5, age_85p * 90])
    pop = sum([age_04, age_59, age_1014, age_1519, age_2024, age_2529,
               age_3034, age_3539, age_4044, age_4549, age_5054, age_5559,
               age_6064, age_6569, age_7074, age_7579, age_8084, age_85p])

    return w_age, pop


def retrieve_cso(data_list, zoom, year, data_dict=None, area='Ireland'):

    if zoom == 'map':
        ed_data = []

        for tl_ed in data_dict.keys():

            with open(settings.BASE_DIR + '/riskdb' + settings.STATIC_URL + 'riskdb/data/cross_ref_dict.txt') as f:
                cross_ref_dict = eval(f.read())
            try:
                cso_ed = cross_ref_dict[tl_ed]
            except:
                continue

            ref_id = CSORef.objects.filter(zoom='edist', desc=cso_ed)[0].uid

            w_age, pop = get_age_stats('edist', ref_id, year)

            oc = Housing.objects.filter(zoom='edist', uid=ref_id, year=year)[0].occupied
            unoc = Housing.objects.filter(zoom='edist', uid=ref_id, year=year)[0].unoccupied

            ed_data.append([w_age, pop, round((oc / (oc + unoc)) * 100, 2), data_dict[tl_ed]])

        count = 0
        w_age = 0
        w_pop = 0
        w_occ = 0
        for i in ed_data:
            count += i[3]
            w_pop += i[1] * i[3]
            w_age += (i[0]/i[1]) * i[3]
            w_occ += i[2] * i[3]

        data_list['dem_age'] = round(w_age/count, 2)

        data_list['population'] = round(w_pop/count, 0)

        data_list['perc_oc'] = round(w_occ/count, 2)

    else:
        ref_id = CSORef.objects.filter(zoom=zoom, desc=area)[0].uid

        w_age, pop = get_age_stats(zoom, ref_id, year)

        data_list['dem_age'] = round(w_age / pop, 2)

        data_list['population'] = pop

        oc = Housing.objects.filter(zoom=zoom, uid=ref_id, year=year)[0].occupied
        unoc = Housing.objects.filter(zoom=zoom, uid=ref_id, year=year)[0].unoccupied

        data_list['perc_oc'] = round((oc / (oc + unoc)) * 100, 2)

    return data_list


def cached_data(request):
    res = {i[0]: i[1] for i in request.GET.items()}
    if res['calcArea'] == 'country':
        with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/country_request.txt') as f:
            country_request = eval(f.read())
        with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/country_request_nobad.txt') as f:
            cr_nobad = eval(f.read())
        if res == country_request:
            with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/country_data.txt') as f:
                country_data = eval(f.read())
                return [True, country_data]
        elif res == cr_nobad:
            with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/country_data_nobad.txt') as f:
                cd_nobad = eval(f.read())
            return [True, cd_nobad]

    elif res['calcArea'] == 'region':
        if res['area'] == 'Leinster':
            with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/leinster_request.txt') as f:
                region_request = eval(f.read())
            with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/leinster_request_nobad.txt') as f:
                rr_nobad = eval(f.read())
            if res == region_request:
                with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/leinster_data.txt') as f:
                    region_data = eval(f.read())
                return [True, region_data]
            elif res == rr_nobad:
                with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/leinster_data_nobad.txt') as f:
                    rd_nobad = eval(f.read())
                return [True, rd_nobad]

        elif res['area'] == 'Munster':
            with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/munster_request.txt') as f:
                region_request = eval(f.read())
            with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/munster_request_nobad.txt') as f:
                rr_nobad = eval(f.read())
            if res == region_request:
                with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/munster_data.txt') as f:
                    region_data = eval(f.read())
                return [True, region_data]
            elif res == rr_nobad:
                with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/munster_data_nobad.txt') as f:
                    rd_nobad = eval(f.read())
                return [True, rd_nobad]


    elif res['calcArea'] == 'county' and res['area'] == 'Dublin':
        with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/dublin_request.txt') as f:
            county_request = eval(f.read())
        with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/dublin_request_nobad.txt') as f:
            cr_nobad = eval(f.read())
        if res == county_request:
            with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/dublin_data.txt') as f:
                county_data = eval(f.read())
            return [True, county_data]
        elif res == cr_nobad:
            with open(settings.BASE_DIR + '/homepage' + settings.STATIC_URL + 'homepage/data/dublin_data_nobad.txt') as f:
                county_data = eval(f.read())
            return [True, county_data]

    return [False, None]


def return_response_stats(request):

    calcArea = request.GET['calcArea']
    if cached_data(request)[0] == True:
        return HttpResponse(json.dumps(cached_data(request)[1]))

    price_low = request.GET['price_low']
    price_high = request.GET['price_high']
    date_low = int(request.GET['date_low'])
    date_high = int(request.GET['date_high'])

    dh = datetime.fromtimestamp(date_high / 1000)
    dl = datetime.fromtimestamp(date_low / 1000)

    goodlist = []

    if calcArea == 'map':
        top = float(request.GET['top'])
        bottom = float(request.GET['bottom'])
        right = float(request.GET['right'])
        left = float(request.GET['left'])

        data = Sale.objects.filter(quality='good', nfma='No').filter(
            latitude__gte=bottom, latitude__lte=top, longitude__gte=left,
            longitude__lte=right, price__lte=price_high, price__gte=price_low,
            sale_date__gte=dl, sale_date__lte=dh)

        if len(data) < 50:
            return HttpResponse(json.dumps({'data': 'Not Enough Data'}))

        ed_count = {}
        for sale in data:
            goodlist.append(sale.uid)
            if sale.ed in ed_count.keys():
                ed_count[sale.ed] += 1
            else:
                ed_count[sale.ed] = 1

        data_list = retrieve_stats(data)

        data_list = retrieve_cso(data_list, 'map', 2016, ed_count)

    elif calcArea == 'radius':
        top = float(request.GET['top'])
        bottom = float(request.GET['bottom'])
        right = float(request.GET['right'])
        left = float(request.GET['left'])

        c_lat = (top - bottom)/2 + bottom
        c_lng = (left - right)/2 + right

        radius = float(request.GET['radius'])
        right = latlng(c_lat, c_lng, radius, 'E')[1]
        left = latlng(c_lat, c_lng, radius, 'W')[1]
        top = latlng(c_lat, c_lng, radius, 'N')[0]
        bottom = latlng(c_lat, c_lng, radius, 'S')[0]


        data = Sale.objects.filter(quality='good', nfma='No').filter(
            latitude__gte=bottom, latitude__lte=top, longitude__gte=left,
            longitude__lte=right, price__lte=price_high, price__gte=price_low,
            sale_date__gte=dl, sale_date__lte=dh)

        if len(data) < 50:
            return HttpResponse("Not Enough Data")

        price_list = []
        date_list = []
        size_list = []

        scatter_data = {'date': [], 'price': []}
        hist_data = []
        ed_count = {}

        for sale in data:
            lat = float(sale.latitude)
            lng = float(sale.longitude)

            if distance(c_lat, c_lng, lat, lng) <= radius:
                goodlist.append(sale.uid)
                raw_time = calendar.timegm(sale.sale_date.timetuple())
                date_list.append(raw_time)
                price_list.append(float(sale.price))
                scatter_data['date'].append(sale.sale_date)
                scatter_data['price'].append(float(sale.price))
                hist_data.append(float(sale.price))
                if sale.ed in ed_count.keys():
                    ed_count[sale.ed] += 1
                else:
                    ed_count[sale.ed] = 1
                if sale.PSD == 'greater than or equal to 38 sq metres and less than 125 sq metres':
                    size_list.append(81.5)
                elif sale.PSD == 'greater than 125 sq metres':
                    size_list.append(125)
                elif sale.PSD == 'less than 38 sq metres':
                    size_list.append(38)

        if len(date_list) == 0:
            return HttpResponse("Not Enough Data")

        df = pd.DataFrame(scatter_data, columns=['price'],
                          index=scatter_data['date'])
        df.index.names = ['date']

        df = df.set_index(pd.DatetimeIndex(df.index))
        df = df.resample('W').mean().dropna(axis=0, how='any')

        scatter_data = list(map(
            lambda x, y: [
                calendar.timegm(x.to_pydatetime().timetuple()) * 1000,
                round(float(y), 2)], df.index.tolist(), df.values))

        compressed_hist_data = compress_list(hist_data, 25000)

        if max(compressed_hist_data.keys()) / 20 < 25000:
            compressed_hist_data = compress_list(hist_data, 20000)
            if max(compressed_hist_data.keys()) / 20 < 20000:
                compressed_hist_data = compress_list(hist_data, 15000)
                if max(compressed_hist_data.keys()) / 20 < 15000:
                    compressed_hist_data = compress_list(hist_data, 10000)

        data_list = {'ave_price': np.mean(price_list),
            'min_price': min(price_list),
            'max_price': max(price_list),
            'min_date': min(date_list) * 1000,
            'max_date': max(date_list) * 1000,
            'med_price': np.median(price_list),
            'avg_date': np.mean(date_list) * 1000,
            'avg_size': round(np.mean(size_list), 1),
            'hist_data': compressed_hist_data,
            'scatter_data': scatter_data}

        data_list = retrieve_cso(data_list, 'map', 2016, ed_count)

    elif calcArea == 'county':
        county = request.GET['area'].capitalize()
        bad_data = request.GET['bad_data']

        if bad_data == 'true':
            data = Sale.objects.filter(nfma='No',
                price__lte=price_high, price__gte=price_low,
                sale_date__gte=dl, sale_date__lte=dh, county=county)
        else:
            data = Sale.objects.filter(quality='good', nfma='No',
                price__lte=price_high, price__gte=price_low,
                sale_date__gte=dl, sale_date__lte=dh, county=county)

        if len(data) < 50:
            return HttpResponse("Not Enough Data")

        for sale in data:
            goodlist.append(sale.uid)

        data_list = retrieve_stats(data)

        data_list = retrieve_cso(data_list, 'county', 2016, area=county)

    elif calcArea == 'region':

        region = request.GET['area'].capitalize()
        bad_data = request.GET['bad_data']

        if bad_data == 'true':
            data = Sale.objects.filter(nfma='No',
                price__lte=price_high, price__gte=price_low,
                sale_date__gte=dl, sale_date__lte=dh, region=region)
        else:
            data = Sale.objects.filter(quality='good', nfma='No',
               price__lte=price_high, price__gte=price_low,
               sale_date__gte=dl, sale_date__lte=dh, region=region)

        if len(data) < 50:
            return HttpResponse("Not Enough Data")

        for sale in data:
            goodlist.append(sale.uid)

        data_list = retrieve_stats(data)

        data_list = retrieve_cso(data_list, 'region', 2016, area=region)

    elif calcArea == 'country':
        bad_data = request.GET['bad_data']

        if bad_data == 'true':
            data = Sale.objects.filter(nfma='No',
                price__lte=price_high, price__gte=price_low,
                sale_date__gte=dl, sale_date__lte=dh)
        else:
            data = Sale.objects.filter(quality='good', nfma='No',
                price__lte=price_high, price__gte=price_low,
                sale_date__gte=dl, sale_date__lte=dh)

        if len(data) < 50:
            return HttpResponse("Not Enough Data")

        for sale in data:
            goodlist.append(sale.uid)

        data_list = retrieve_stats(data)

        data_list = retrieve_cso(data_list, 'country', 2016)

    data_list['shade'] = goodlist

    data_list['lobf_coef'] = list(np.polyfit([i[0] for i in data_list['scatter_data']], [i[1] for i in data_list['scatter_data']], deg = 3))

    return HttpResponse(json.dumps(data_list))


def parse_tf(tf):
    if tf == 'true':
        return True
    else:
        return False


def write_error(request):

    error = ReportedErrors()

    error.marker_uid = request.POST['uid']
    error.address_error = parse_tf(request.POST['address-error'])
    error.location_error = parse_tf(request.POST['location-error'])
    error.date_error = parse_tf(request.POST['date-error'])
    error.price_error = parse_tf(request.POST['price-error'])

    other_info = request.POST['text']

    if len(other_info) > 500:
        other_info = other_info[:500]

    error.other_info = other_info

    error.save()

    return HttpResponse('Thank you')