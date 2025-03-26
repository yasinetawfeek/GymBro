
import os
import django

"""
To avoid the following error keep the below statement

backend-1           |     raise ImproperlyConfigured(
backend-1           | django.core.exceptions.ImproperlyConfigured: Requested setting INSTALLED_APPS, but settings are not configured. You must either define the environment variable DJANGO_SETTINGS_MODULE or call settings.configure() 
before accessing settings.
"""
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from django.contrib.auth.models import User
from django.contrib.auth.models import Group, Permission
from django.contrib.contenttypes.models import ContentType


"""
Create Groups
1-Admin
2-AI Engineer
3-Customer

"""


query = User.objects.all()
query.delete()

content_type = ContentType.objects.get(app_label='auth', model='user')

machine_learning_permission, created = Permission.objects.get_or_create(
    codename='machine_learning_permission',
    name='Can Train Machine Learning Models',
    content_type=content_type,
)


admin_group, created = Group.objects.get_or_create(name='Admin')
all_permissions = Permission.objects.all()
admin_group.permissions.add(*all_permissions)


ai_engineer_group, created = Group.objects.get_or_create(name='AI Engineer')
ai_engineer_group.permissions.add(machine_learning_permission)
ai_engineer_group.save()
customer_group, created = Group.objects.get_or_create(name='Customer')



staff_user = User.objects.create_user(username='Dong2025', password='Dong2025')




ChatGPT = User.objects.create_user(username='ChatGPT', password='ChatGPT')
ChatGPT.groups.add(ai_engineer_group)
ChatGPT.save()

# Set them as staff
staff_user.groups.add(admin_group)
staff_user.save()

staff_two = User.objects.create_user(username='Yahia2025', password='Yahia2025')
staff_two.groups.add(admin_group)

staff_two.save()

# """
# creating a bunch of users [customer role]

# """

for i in range(1, 20):
    user = User.objects.create_user(username=f'user_{i}', password=f'user_{i}')
    print(f'User {i} created successfully')
    user.groups.add(customer_group)
    user.save()



users = User.objects.filter(groups=admin_group, is_staff=False)
for user in users:
    user.is_staff = True
    user.save()




permissions = ai_engineer_group.permissions.all()

# Print the permissions
for permission in permissions:
    print(f"Permission: {permission.codename} - {permission.name}")