var current_markers = [];
var current_clusters = [];

function update_markers(){
    // Removes all current markers from map
    while(current_markers.length > 0) {
        current_markers.pop().setMap(null);
    }

    // Removes all current clusters from map
    // if (markerCluster.aAddClusterIcons != null){
    //     while(current_clusters.length > 0){
    //         a = current_clusters.pop();
    //         markerCluster.RemoveClusters(a[0], a[1], a[2]);
    //     }
    // }

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

function initMap() {

    map = new google.maps.Map(document.getElementById('map-area'), {
        center: {lat: 53.296704, lng: -6.183750},
        zoom: 13,
        gestureHandling: 'cooperative',
        scrollwheel: false
    });

    google.maps.event.addListener(map, 'idle', function () {
        console.log('idle');
        update_markers();
    });
}