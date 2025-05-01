from django.core.management.base import BaseCommand
from django.db import connection

class Command(BaseCommand):
    help = 'Clean up duplicate ML model entries using raw SQL'

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING('Checking for duplicate ML models...'))
        
        # Display all models first
        with connection.cursor() as cursor:
            cursor.execute("SELECT id, name, model_type, deployed FROM DESD_App_mlmodel")
            rows = cursor.fetchall()
            
            self.stdout.write(self.style.WARNING('Current models in database:'))
            for row in rows:
                self.stdout.write(f"ID: {row[0]}, Name: {row[1]}, Type: {row[2]}, Deployed: {row[3]}")
            
            model_count = len(rows)
            self.stdout.write(self.style.WARNING(f'Total models before cleanup: {model_count}'))
            
            # Delete models with IDs 1, 2, and 3
            cursor.execute("DELETE FROM DESD_App_mlmodel WHERE id IN (1, 2, 3)")
            deleted_count = cursor.rowcount
            
            self.stdout.write(self.style.SUCCESS(f'Deleted {deleted_count} models directly with SQL'))
            
            # Check remaining models
            cursor.execute("SELECT id, name, model_type, deployed FROM DESD_App_mlmodel")
            remaining_rows = cursor.fetchall()
            
            self.stdout.write(self.style.WARNING('Remaining models in database:'))
            for row in remaining_rows:
                self.stdout.write(f"ID: {row[0]}, Name: {row[1]}, Type: {row[2]}, Deployed: {row[3]}")
            
            self.stdout.write(self.style.SUCCESS(f'Total models after cleanup: {len(remaining_rows)}')) 