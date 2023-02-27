import pickle
from pathlib import Path


import streamlit-authenticator as stauth

names = ['Peter Parker', 'Bella Porch']
usernames = ['pparker', 'bporch']
passwords = ['qwerty', '12345']

hashed_passwords = stauth.Hasher(passwords).generate()

file_path = Path(__file__).parent / 'hashed_pw.pkl'
with file_path.open('wb') as file:
    pickle.dump(hashed_passwords, file)

