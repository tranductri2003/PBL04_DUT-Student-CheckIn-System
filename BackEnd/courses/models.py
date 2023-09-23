from django.db import models
from django.utils import timezone
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from users.models import NewUser
from helper.models import TimeSetup

DAY_CHOICES = [
    (2, _('Monday')),
    (3, _('Tuesday')),
    (4, _('Wednesday')),
    (5, _('Thursday')),
    (6, _('Friday')),
    (7, _('Saturday')),
    (8, _('Sunday')),
]
# Create your models here.
class Courses(TimeSetup, models.Model):
    course_id = models.CharField(max_length=15, unique=True)
    course_name = models.CharField(max_length=50)
    teacher_id = models.ForeignKey(
        NewUser, on_delete=models.CASCADE)
    num_of_student = models.IntegerField(default=0)
    day_of_week = models.IntegerField(choices=DAY_CHOICES, default=1)
    start_time = models.TimeField(default=timezone.now)  # Sử dụng TimeField
    end_time = models.TimeField(default=timezone.now)
    start_date = models.DateField(default=timezone.now)  # Sử dụng TimeField
    end_date = models.DateField(default=timezone.now)
    room = models.CharField(max_length=50)

    def duration(self):
        return self.end_time - self.start_time
    def __str__(self):
        return self.course_name