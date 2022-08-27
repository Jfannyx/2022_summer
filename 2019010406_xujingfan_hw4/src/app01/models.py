from django.db import models

class task(models.Model):
    id = models.AutoField(primary_key=True, verbose_name="ID")
    task_name = models.CharField(max_length=20)
    username = models.CharField(max_length=20)
    init_time = models.DateTimeField(auto_now_add=True)
    end_time = models.DateTimeField(auto_now=True)
    status = models.BooleanField(max_length=20,default=False)
