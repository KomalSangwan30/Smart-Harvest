from django.db import models

# This model saves each prediction to the database
class Prediction(models.Model):
    # Input values (soil and weather conditions)
    nitrogen    = models.FloatField()
    phosphorus  = models.FloatField()
    potassium   = models.FloatField()
    temperature = models.FloatField()
    humidity    = models.FloatField()
    ph          = models.FloatField()
    rainfall    = models.FloatField()

    # Output (what the ML model predicted)
    crop_name   = models.CharField(max_length=100)
    confidence  = models.FloatField(null=True, blank=True)  # how confident the model is

    # Auto-saved timestamp
    created_at  = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.crop_name} - {self.created_at.strftime('%d %b %Y')}"
