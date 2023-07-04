from flask import Flask, render_template ,request ,redirect ,flash ,session
import subprocess
import sqlite3
import os
import csv
import hashlib

os.chdir("/home/user/Downloads/DDDDeep")

app = Flask(__name__, template_folder='.')
app.secret_key = 'my_secret_key'

# Connect to SQLite database
conn = sqlite3.connect('users.db', check_same_thread=False)
c = conn.cursor()

# Create users table if it doesn't already exist
c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY AUTOINCREMENT,
              username TEXT NOT NULL UNIQUE,
              password TEXT NOT NULL)''')
conn.commit()

@app.route('/')
def home():
    return render_template('login_form.html')

@app.route('/submit_form', methods=['GET', 'POST'])
def submit_form():
    if request.method == 'POST':
        # Get username and password from form submission
        username = request.form['username']
        password = request.form['password']

        # Hash password using SHA256 algorithm
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Check if user exists in database
        c.execute('SELECT * FROM users WHERE username=? AND password=?', (username, hashed_password))
        user = c.fetchone()

        if user:
            # Save user session and redirect to dashboard
            session['user_id'] = user[0]
            session['username'] = user[1]
            return redirect('/dashboard')
        elif username == 'admin' and password == 'admin':
            return redirect('/dashboard')
        else:
            flash('Invalid username or password!')
            return redirect('/submit_form')

    return render_template('login_form.html')

@app.route('/dashboard')
def dashboard():
    # Redirect to login page if user is not logged in
    if 'user_id' not in session:
        return render_template('login_form.html', error='Invalid login credentials')
    else:
        return render_template('index1.html')
    
'''
@app.route('/submit_form', methods=['POST'])
def submit_form():
    username = request.form['username']
    password = request.form['password']

    # Perform validation on the submitted username and password
    if username == 'my_username' and password == 'my_password':
        return redirect('/dashboard')
    else:
        return render_template('login_form.html', error='Invalid login credentials')


@app.route('/dashboard')
def dashboard():
    return render_template('index1.html')


@app.route('/')
def index():
    return render_template('index1.html')

'''

@app.route('/register',methods=['POST'])
def register():
    return render_template('register.html')

@app.route('/register_form', methods=['GET', 'POST'])
def register_form():
    if request.method == 'POST':
        # Get username and password from form submission
        username = request.form['username']
        password = request.form['password']

        # Hash password using SHA256 algorithm
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        try:
            # Insert user into database
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            flash('User registered successfully!')
            return redirect('/submit_form')
        except sqlite3.IntegrityError:
            flash('Username already exists!')
            return redirect('/register_form')

    return render_template('register.html')

@app.route('/run-script')
def run_script():
    subprocess.Popen(['/usr/bin/python3','/home/user/Downloads/DDDDeep/main4a.py'])
    return "Please Wait.... Script is running."

@app.route('/database')
def database():
    data = []
    with open('/home/user/Downloads/DDDDeep/EY_Dataset/dataset_new/data.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    return render_template('table.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)

