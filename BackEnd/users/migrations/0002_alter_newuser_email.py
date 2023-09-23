# Generated by Django 4.2.5 on 2023-09-13 05:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='newuser',
            name='email',
            field=models.EmailField(default='admin@gmail.com', max_length=254, unique=True, verbose_name='email address'),
        ),
    ]
