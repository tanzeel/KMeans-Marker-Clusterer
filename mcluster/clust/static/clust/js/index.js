var current_markers = [];
var current_clusters = [];
var markerCluster;

function ajax_update_markers(){
    var r = map.getBounds().getNorthEast().lng();
    var l = map.getBounds().getSouthWest().lng();
    var t = map.getBounds().getNorthEast().lat();
    var b = map.getBounds().getSouthWest().lat();

    $.ajax({
        type: "GET",
        url: "/ajax-response-markers/",
        data: {'right': r, 'left': l,
            'top': t, 'bottom': b, 'zoom': map.getZoom()},
        async: true,
        dataType: 'json',
        success: function (data) {
            plotPoints(map, data);
        }
    });
}

function plotPoints(map, data){
    while(current_markers.length > 0) {
        current_markers.pop().setMap(null);
    }

    if (markerCluster.aAddClusterIcons != null){
        while(current_clusters.length > 0){
            a = current_clusters.pop();
            markerCluster.RemoveClusters(a[0], a[1], a[2]);
        }
    }

    if (data['cluster'] == 'true'){
        cluster_data = data['data'];
        for (var i = 0; i < cluster_data.length; i++){
            if (data['data'][i][2] != 0) {
                markerCluster.AddCluster(cluster_data[i][0], cluster_data[i][1], cluster_data[i][2]);
                current_clusters.push([cluster_data[i][0], cluster_data[i][1], cluster_data[i][2]])
            }
        }
    } else {
        for(var i = 0; i < data.length; i++) {
            var lat = parseFloat(data[i]['fields']['latitude']);
            var lng = parseFloat(data[i]['fields']['longitude']);

            var pos = new google.maps.LatLng(lat, lng);

            var marker = new google.maps.Marker({
                position: pos,
                map: map
            });

            current_markers.push(marker)
        }
    }
}


function update_markers(){
    // Removes all current markers from map
    while(current_markers.length > 0) {
        current_markers.pop().setMap(null);
    }

    for(var i = 0; i < cluster_data.length; i++) {
        var lat = parseFloat(cluster_data[i][0]);
        var lng = parseFloat(cluster_data[i][1]);

        var pos = new google.maps.LatLng(lat, lng);

        var marker = new google.maps.Marker({
            position: pos,
            map: map
        });

        current_markers.push(marker)
    }
}