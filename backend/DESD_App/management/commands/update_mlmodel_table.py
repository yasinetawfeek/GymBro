from django.core.management.base import BaseCommand
from django.db import connection

class Command(BaseCommand):
    help = 'Updates the MLModel table to match the current model definition'

    def handle(self, *args, **options):
        self.stdout.write(self.style.WARNING('Checking MLModel table structure...'))
        
        # First, we check if the table exists and get its current structure
        with connection.cursor() as cursor:
            cursor.execute("PRAGMA table_info(DESD_App_mlmodel)")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            self.stdout.write(f"Current columns: {', '.join(column_names)}")
            
            # We need to make the table match our model
            # This requires creating a new table with the correct schema and copying data
            
            # Step 1: Create a new table with the correct schema
            self.stdout.write(self.style.WARNING('Creating new table with correct schema...'))
            
            try:
                # Create a temporary table with the correct schema
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS DESD_App_mlmodel_new (
                    id integer NOT NULL PRIMARY KEY AUTOINCREMENT,
                    name varchar(255) NOT NULL,
                    model_type varchar(32) NOT NULL,
                    learning_rate real NOT NULL DEFAULT 0.001,
                    epochs integer NOT NULL DEFAULT 100,
                    batch_size integer NOT NULL DEFAULT 32,
                    accuracy real NOT NULL DEFAULT 0.0,
                    deployed boolean NOT NULL DEFAULT 0,
                    created_at datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_updated datetime NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    last_trained date NULL
                );
                """)
                self.stdout.write(self.style.SUCCESS('Created new table structure'))
                
                # Step 2: Copy data from old table to new table, mapping fields correctly
                try:
                    # Figure out which fields exist in the old table
                    old_fields = []
                    new_fields = ['id', 'name', 'model_type']
                    field_mappings = {
                        'version': None,  # We don't need this in the new schema
                        'description': None,  # We don't need this in the new schema
                        'updated_at': 'last_updated',  # Map old name to new name
                    }
                    
                    for old_col in column_names:
                        if old_col in ['id', 'name', 'model_type']:
                            old_fields.append(old_col)
                        elif old_col in field_mappings and field_mappings[old_col] is not None:
                            old_fields.append(old_col)
                            new_fields.append(field_mappings[old_col])
                        elif old_col in ['learning_rate', 'epochs', 'batch_size', 'accuracy', 
                                         'deployed', 'created_at', 'last_updated', 'last_trained']:
                            old_fields.append(old_col)
                            new_fields.append(old_col)
                    
                    # Insert statement with correct field mapping
                    old_fields_sql = ', '.join(old_fields)
                    new_fields_sql = ', '.join(new_fields)
                    
                    # Insert command with mapped fields
                    if old_fields:
                        cursor.execute(f"""
                        INSERT INTO DESD_App_mlmodel_new ({new_fields_sql})
                        SELECT {old_fields_sql} FROM DESD_App_mlmodel;
                        """)
                        self.stdout.write(self.style.SUCCESS('Copied data to new table'))
                    else:
                        self.stdout.write(self.style.WARNING('No data to copy'))
                    
                    # Step 3: Drop the old table and rename the new one
                    cursor.execute("DROP TABLE DESD_App_mlmodel;")
                    cursor.execute("ALTER TABLE DESD_App_mlmodel_new RENAME TO DESD_App_mlmodel;")
                    self.stdout.write(self.style.SUCCESS('Renamed new table to replace old one'))
                    
                    # Step 4: Create necessary indexes
                    cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS DESD_App_mlmodel_unique_deployed
                    ON DESD_App_mlmodel (model_type, deployed)
                    WHERE deployed=1;
                    """)
                    self.stdout.write(self.style.SUCCESS('Created unique constraint for deployed models'))
                    
                    self.stdout.write(self.style.SUCCESS('Table structure updated successfully!'))
                    
                except Exception as e:
                    self.stdout.write(self.style.ERROR(f'Error during data migration: {e}'))
                    
            except Exception as e:
                self.stdout.write(self.style.ERROR(f'Error creating new table: {e}')) 