from deta import Deta
import os
from dotenv import load_dotenv
import bcrypt
import io
import pandas as pd

# load the environment variables
load_dotenv('.env')
DETA_KEY = os.getenv("DETA_KEY")

# Initialize with a project key
deta = Deta(DETA_KEY)

# Connection to the database
db = deta.Base('AuthForAutoML')
files_db = deta.Base('user-files')

# Connection to the drive
drive = deta.Drive("csv-files")

def insert_user(username, name, password):
    """Return the used on successful user creation, otherwise raises an error"""
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt)
    return db.put({"key": username, "name": name, "password":  hashed_password.decode('utf-8'), "salt": salt.decode('utf-8')})

def fetch_all_users():
    """Returns a dict of all users"""
    users = db.fetch()
    return users.items

def get_user(username):
    """If not found, the function will return None"""
    return db.get(username)

def update_user(username, updates):
    """If the item is updated, returns None. Otherwise, an exception is raised"""
    return db.update(updates, username)

def delete_user(username):
    """Always returns None, even if the key does not exist"""
    return db.delete(username)

def upload_file(username, df, filename):
    """Upload a file to the user's database"""
    csv_data = df.to_csv(index=False)

    drive.put(filename, csv_data.encode())

    files_db.put({
        'user': username,
        'file_name': filename
    })

def get_file(filename):
    """Retrieve a file from the user's database"""
    csv_data = drive.get(filename).content.decode()
    df = pd.read_csv(io.StringIO(csv_data))
    return df

