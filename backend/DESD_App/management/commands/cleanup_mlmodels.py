from django.core.management.base import BaseCommand
from DESD_App.models import MLModel

class Command(BaseCommand):
    help = 'Clean up duplicate ML model entries'

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING('Checking for duplicate ML models...'))
        
        # Display all models first
        self.stdout.write(self.style.WARNING('Current models in database:'))
        for model in MLModel.objects.all():
            self.stdout.write(f"ID: {model.id}, Name: {model.name}, Type: {model.model_type}, Deployed: {model.deployed}")
        
        # Count before
        count_before = MLModel.objects.count()
        self.stdout.write(self.style.WARNING(f'Total models before cleanup: {count_before}'))
        
        # The proper models that we want to keep (based on ID)
        proper_model_ids = [4, 5, 6]  # IDs of WorkoutClassifier-v1, PoseCorrectionNet-v1, MuscleActivationModel-v1
        
        # Delete all other models
        other_models = MLModel.objects.exclude(id__in=proper_model_ids)
        count = other_models.count()
        if count > 0:
            other_models.delete()
            self.stdout.write(self.style.SUCCESS(f'Deleted {count} duplicate models'))
        
        # Count after
        count_after = MLModel.objects.count()
        self.stdout.write(self.style.SUCCESS(f'Total models after cleanup: {count_after}'))
        self.stdout.write(self.style.SUCCESS(f'Deleted {count_before - count_after} duplicate models'))
        
        # Display remaining models
        self.stdout.write(self.style.WARNING('Remaining models in database:'))
        for model in MLModel.objects.all():
            self.stdout.write(f"ID: {model.id}, Name: {model.name}, Type: {model.model_type}, Deployed: {model.deployed}") 