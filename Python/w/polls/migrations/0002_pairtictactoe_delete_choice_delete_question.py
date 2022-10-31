# Generated by Django 4.1.1 on 2022-10-25 23:14

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='PairTicTacToe',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('room_code', models.CharField(max_length=100)),
                ('game_creator', models.CharField(max_length=100)),
                ('game_opponent', models.CharField(blank=True, max_length=100, null=True)),
                ('is_over', models.BooleanField(default=False)),
            ],
        ),
        migrations.DeleteModel(
            name='Choice',
        ),
        migrations.DeleteModel(
            name='Question',
        ),
    ]