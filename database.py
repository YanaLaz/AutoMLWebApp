from deta import Deta
import os
from dotenv import load_dotenv

# load the environment variables
load_dotenv('.env')
DETA_KEY = os.getenv("DETA_KEY")

# Initialize with a project key
deta = Deta(DETA_KEY)

# Connection to the database
db = deta.Base('AuthForAutoML')

def insert_user(username, name, password):
    """Return the used on successful user creation, otherwise raises an error"""
    return db.put({"key": username, "name": name, "password": password})

# insert_user('yanix','Yana','qwerty')

def fetch_all_users():
    """Returns a dict of all users"""
    users = db.fetch()
    return users.items

# print(fetch_all_users())

def get_user(username):
    """If not found, the function will return None"""
    return db.get(username)

# print(get_user('yanix'))

def update_user(username, updates):
    """If the item is updated, returns None. Otherwise, an exception is raised"""
    return db.update(updates, username)

# update_user('yanix', updates={'name': "Yana Lazareva"})

def delete_user(username):
    """Always returns None, even if the key does not exist"""
    return db.delete(username)

# delete_user('yanix')


