#!/bin/sh

#this program is written in Bash, which is a shell scripting language like fish, or zsh

echo "Waiting for PostgreSQL to start..."

./Wait-for "$DB_HOST":"$DB_PORT"

echo "Executing manage.py"

# Check if we should reset the database before running the script
if [ "$RESET_DB" = "true" ]; then
    echo "Resetting database as requested by RESET_DB flag"
    
    # Create migrations if needed before attempting any operations
    python manage.py makemigrations
    
    # Connect to PostgreSQL and drop/recreate the database
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -c "DROP SCHEMA public CASCADE;"
    PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -U $DB_USER -c "CREATE SCHEMA public;"
    
    # Apply migrations one app at a time in proper order
    python manage.py migrate auth
    python manage.py migrate contenttypes
    python manage.py migrate admin
    python manage.py migrate sessions
    python manage.py migrate DESD_App 0001_initial
    python manage.py migrate DESD_App 0002
    python manage.py migrate DESD_App 0003
    python manage.py migrate DESD_App 0004
    python manage.py migrate DESD_App 0005
    python manage.py migrate DESD_App 0006
    python manage.py migrate DESD_App 0007
    python manage.py migrate DESD_App
    
    # Clean up any remaining User objects if needed
    python manage.py shell -c "from django.contrib.auth.models import User; User.objects.all().delete()"
else
    # Create migrations if needed
    python manage.py makemigrations

    # Apply migrations
    python manage.py migrate
fi

# Run the initialization script with error handling
python manage.py shell -c "
import os
import sys
try:
    exec(open('automatically_start_script.py').read())
    print('Initialization script completed successfully')
except Exception as e:
    if 'UNIQUE constraint failed' in str(e):
        print('Users already exist in the database. Skipping user creation.')
    else:
        print(f'Error running initialization script: {e}')
        sys.exit(1)
"

echo "Starting server..."
python manage.py runserver 0.0.0.0:8000 #running the server


