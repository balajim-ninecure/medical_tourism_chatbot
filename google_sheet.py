import gspread
from google.oauth2.service_account import Credentials

scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_file("ninecure-chatbot-498509-75f1ae3bfc05.json",scopes = scopes)

client = gspread.authorize(creds)

sheet = client.open("ninecure lead").sheet1