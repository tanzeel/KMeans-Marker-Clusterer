from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.index, name='index'),
    url(r'^ajax-response-markers/$', views.return_response_markers,
        name='return_response_markers'),
]
