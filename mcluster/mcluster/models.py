from django.db import models

ZOOM_CHOICES = (
        ('region', 'region'), ('country', 'country'), ('county', 'county'),
        ('edist', 'edist'), ('area', 'area'))

# Create your models here.
class Sale(models.Model):
    uid = models.IntegerField()
    sale_date = models.DateField()

    address = models.CharField(max_length=500)
    postcode = models.CharField(max_length=50)

    COUNTY_CHOICES = (
        ('Galway','Galway'), ('Leitrim','Leitrim'), ('Mayo','Mayo'),
        ('Roscommon','Roscommon'), ('Sligo','Sligo'), ('Carlow','Carlow'),
        ('Dublin','Dublin'), ('Kildare','Kildare'), ('Kilkenny','Kilkenny'),
        ('Laois','Laois'), ('Longford','Longford'), ('Louth','Louth'),
        ('Meath','Meath'), ('Offaly','Offaly'), ('Westmeath','Westmeath'),
        ('Wexford','Wexford'), ('Wicklow','Wicklow'), ('Clare','Clare'),
        ('Cork','Cork'), ('Kerry','Kerry'), ('Limerick','Limerick'),
        ('Tipperary','Tipperary'), ('Waterford','Waterford'),
        ('Cavan','Cavan'), ('Donegal','Donegal'), ('Monaghan','Monaghan')
    )

    county = models.CharField(max_length=9, choices=COUNTY_CHOICES,
                              default='Dublin')
    price = models.DecimalField(decimal_places=2, max_digits=10)
    nfma = models.CharField(max_length=3, choices=(('yes', 'yes'),
                            ('no', 'no')), default='no')
    vat_ex = models.CharField(max_length=3, choices=(('yes', 'yes'),
                            ('no', 'no')), default='no')
    DoP_CHOICES = (
        ('Second-Hand Dwelling house /Apartment','Second-Hand'),
        ('New Dwelling house /Apartment','New')
    )

    DoP = models.CharField(max_length=40, choices=DoP_CHOICES,
                              default='Second-Hand Dwelling house /Apartment')

    PSD_CHOICES = (
        ('greater than or equal to 38 sq metres and less than 125 sq '
         'metres', 'GT38 LS125'),
        ('greater than 125 sq metres','GT125'),
        ('less than 38 sq metres', 'LT38'),('Unknown', 'Unknown')
    )

    PSD = models.CharField(max_length=70, choices=DoP_CHOICES,
                           default='Unknown')

    REGION_CHOICES = (
        ('Connacht','Connacht'), ('Leinster', 'Leinster'),
        ('Munster', 'Munster'), ('Ulster', 'Ulster'))

    region = models.CharField(max_length=8,choices=REGION_CHOICES,
                              default='Leinster')
    latitude = models.DecimalField(max_digits=9,decimal_places=7)
    longitude =  models.DecimalField(max_digits=10,decimal_places=7)

    ed = models.CharField(max_length=100)

    def __str__(self):
        return str(self.id)