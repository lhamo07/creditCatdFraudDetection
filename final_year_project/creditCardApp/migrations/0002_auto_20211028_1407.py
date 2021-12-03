# Generated by Django 3.2.8 on 2021-10-28 08:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('creditCardApp', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='fileUpload',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('photo', models.FileField(upload_to='')),
            ],
            options={
                'verbose_name_plural': 'Puppies',
            },
        ),
        migrations.DeleteModel(
            name='LoginForm',
        ),
    ]