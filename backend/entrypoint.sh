#!/bin/sh

#this program is written in Bash, which is a shell scripting language like fish, or zsh

echo "Waiting for PostgreSQL to start..."

./Wait-for "$DB_HOST":"$DB_PORT"

echo "Executing manage.py"

# Check if we should reset the database before running the script
if [ "$RESET_DB" = "true" ]; then
    echo "Resetting database as requested by RESET_DB flag"
    # For PostgreSQL, we need to drop and recreate tables properly
    # First, make sure migrations are created
    python manage.py makemigrations
    
    # Use Django's flush command to clear the database
    python manage.py flush --no-input
    
    # Apply all migrations from scratch in the correct order
    python manage.py migrate --no-input
    
    # Delete all users if needed
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


