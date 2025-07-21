# models.py
from django.db import models
from django.contrib.auth.models import User

class MalariaPrediction(models.Model):
    DISTRICT_CHOICES = [
        ('CHAKE-CHAKE', 'Chake-Chake'),
        ('KASKAZINI A', 'Kaskazini A'),
        ('KASKAZINI B', 'Kaskazini B'),
        ('KATI', 'Kati'),
        ('KUSINI', 'Kusini'),
        ('MAGHARIBI A', 'Magharibi A'),
        ('MAGHARIBI B', 'Magharibi B'),
        ('MICHEWENI', 'Micheweni'),
        ('MJINI', 'Mjini'),
        ('MKOANI', 'Mkoani'),
        ('WETE', 'Wete'),
    ]
    
    district = models.CharField(max_length=50, choices=DISTRICT_CHOICES)
    year = models.IntegerField()
    month = models.IntegerField()
    week = models.IntegerField()
    prediction = models.FloatField()  # Stores the probability or risk score
    date_predicted = models.DateTimeField(auto_now_add=True)
    
    # Other fields from your form can be added here as needed
    gender = models.CharField(max_length=10, null=True, blank=True)
    age_group = models.CharField(max_length=10, null=True, blank=True)
    humidity = models.FloatField(null=True, blank=True)
    rainfall = models.FloatField(null=True, blank=True)
    min_temperature = models.FloatField(null=True, blank=True)
    max_temperature = models.FloatField(null=True, blank=True)
    
    class Meta:
        unique_together = ('district', 'year', 'month', 'week')  # Ensures one prediction per district per week
    
    def __str__(self):
        return f"{self.district} - {self.year}-{self.month}-{self.week}: {self.prediction}"