from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
import cv2
#from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer
import pickle
import base64
from PIL import Image
import io
import pickle
# Route for uploading profile photo
from flask import request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import matplotlib.pyplot as plt
load_dotenv()

# Use environment variables to connect to the database
db_host = os.getenv('DB_HOST')
db_database = os.getenv('DB_DATABASE')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')



app = Flask(__name__)
app.config['SECRET_KEY'] = 'abcde1234'

#mail = Mail(app)
serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])

# Load the trained model
model = load_model('skin_cancer_oversampling_model.keras')
model_img = load_model('skin_cancer_image_only_model.keras')

# Define the upload directory
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS




# Route for the prediction page within the user area
@app.route('/prediction')
def prediction():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    return render_template('user_prediction.html')

def connect_db():
    try:
        conn = mysql.connector.connect(
            host=db_host',
            database=db_database,
            user=db_user,
            password=db_password
        )
        if conn.is_connected():
            print('Connected to MySQL database')
            return conn
    except Error as e:
        print(f'Error connecting to MySQL database: {e}')
        raise  # Raise the error to indicate that the connection failed


# Route to process form data
@app.route('/process_input', methods=['POST'])
def process_input():
    try:
        # Get form data
        age = int(request.form['age'])
        sex = request.form['sex']
        anatomical_site = request.form['anatomical_site']
        image_file = request.files['image']

        # Make prediction
        prediction, interpretation, image = make_prediction(age, sex, anatomical_site, image_file)
        
        # Insert the original image data into the database
        original_image= original_image_data(image_file,target_size=(50, 50))

        # Save prediction data to the database
        user_id = session.get('user_id')  # Get the user_id from the session
        save_prediction_to_db(user_id, image, prediction, anatomical_site, original_image)
        
        # Return result to user
        return render_template('user_prediction_result.html', prediction=prediction, interpretation=interpretation)
    except Exception as e:
        # Exception handling
        print(f"An error occurred: {e}")
        return render_template('error.html', message='An error occurred while processing your request.')


# Define the function to load and preprocess the image
def load_and_preprocess_image(image_file):
    img = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (50, 50))
    return img_resized.astype(np.float32) / 255.0

# Function to make predictions based on user input
anatomical_site_features = {
    'anterior torso': 'anatomical_site_anterior torso',
    'head/neck': 'anatomical_site_head/neck',
    'lateral torso': 'anatomical_site_lateral torso',
    'lower extremity': 'anatomical_site_lower extremity',
    'oral/genital': 'anatomical_site_oral/genital',
    'palms/soles': 'anatomical_site_palms/soles',
    'posterior torso': 'anatomical_site_posterior torso',
    'upper extremity': 'anatomical_site_upper extremity'
}

def make_prediction(age, sex, anatomical_site, image_file):
    # Define features dictionary
    features = {
        'age': age,
        'sex_male': 1 if sex == 'male' else 0,
        'sex_female': 1 if sex == 'female' else 0,
        **{value: 1 if anatomical_site == key else 0 for key, value in anatomical_site_features.items()}
    }

    # Load and preprocess image
    image = load_and_preprocess_image(image_file)
    # Make predictions
    predictions = model.predict([np.expand_dims(image, axis=0), np.array([list(features.values())])])

    # Define the decision threshold
    threshold = 0.5 

    # Apply the decision threshold to predictions
    prediction = 'Suspicious lesion' if predictions[0][0] > threshold else 'Benign'

    # Contextual interpretation
    interpretation = ( "Based on the model's prediction, the skin lesion is classified as a suspicious lesion. "
                        "Immediate medical guidance is recommended for further evaluation and possible treatment."
       
    ) if prediction == 'Suspicious lesion' else (
        "Based on the model's prediction, the skin lesion is classified as a  benign. "
        "Monitoring the lesion and consulting a doctor if there are significant changes is advised."
    )

    return prediction, interpretation, image


def resize_image(image, target_size=(50, 50)):
    try:
        # Redimensiona a imagem para o tamanho alvo
        resized_img = cv2.resize(image, target_size)
        
        return resized_img
    
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

def original_image_data(image_object, target_size=(50, 50)):
    try:
        # Verifica se o argumento é um objeto FileStorage
        if hasattr(image_object, 'read'):
            # Se sim, verifica se os dados não estão vazios
            if image_object.seekable():
                image_object.seek(0, 2)
                size = image_object.tell()
                image_object.seek(0)
                if size > 0:
                    # Lê os bytes do objeto
                    image_bytes = image_object.read()
                    
                    # Redimensiona a imagem
                    image = np.frombuffer(image_bytes, dtype=np.uint8)
                    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
                    resized_image = resize_image(image, target_size)
                    
                    # Converte a imagem redimensionada de volta para bytes
                    _, resized_image_bytes = cv2.imencode('.jpg', resized_image)
                    
                    return resized_image_bytes.tobytes()
                else:
                    print("Error: O objeto de imagem está vazio.")
                    return None
            else:
                print("Error: O objeto de imagem não é seekable.")
                return None
        else:
            # Caso contrário, assume que é um caminho de arquivo e lê os bytes do arquivo
            with open(image_object, 'rb') as f:
                image_bytes = f.read()
        
        return image_bytes

    except Exception as e:
        print(f"Error reading image data: {e}")
        return None




# Function to save image and prediction data to the database

def save_prediction_to_db(user_id, image, prediction, anatomical_site, original_image):
    conn = connect_db()
    cursor = conn.cursor()
    try:
        image_data_base64 = pickle.dumps(image)
        cursor.execute("INSERT INTO predictions (user_id, image_data, prediction_result, anatomical_site, original_images) VALUES (%s, %s, %s, %s,%s)", (user_id, image_data_base64, prediction, anatomical_site, original_image))
        
        # Commit the transaction
        conn.commit()
        
        print("Prediction data saved to database")
    except Error as e:
        print(f"Error saving prediction data to database: {e}")
    finally:
        # Close the cursor and connection
        cursor.close()
        conn.close()

def get_selected_prediction(prediction_id):
    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            sql = "SELECT id, anatomical_site, prediction_result, image_data, created_at, original_images FROM predictions WHERE id = %s"
            cursor.execute(sql, (prediction_id,))
            prediction = cursor.fetchone()

            if prediction:
                prediction_id, anatomical_site, prediction_result, image_data_bytes, created_at, original_image_bytes = prediction
                original_image_base64 = base64.b64encode(original_image_bytes).decode('utf-8')
                return {
                    'id': prediction_id,
                    'anatomical_site': anatomical_site,
                    'prediction_result': prediction_result,
                    'image_data': image_data_bytes,
                    'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'original_image_base64': original_image_base64,
                }
            else:
                print("Prediction not found.")
                return None
    except Exception as e:
        print(f"Error fetching prediction: {e}")
        return None
    finally:
        conn.close()

def get_last_prediction(user_id):
    try:
        conn = connect_db()
        with conn.cursor() as cursor:
            sql = "SELECT anatomical_site, prediction_result, image_data, created_at, original_images FROM predictions WHERE user_id = %s ORDER BY created_at DESC LIMIT 1"
            cursor.execute(sql, (user_id,))
            last_prediction = cursor.fetchone()

            if last_prediction:
                anatomical_site, prediction_result, image_data_bytes, created_at, original_image_bytes = last_prediction
                
                # Convertendo a imagem original para base64
                original_image_base64 = base64.b64encode(original_image_bytes).decode('utf-8')

                return {
                    'anatomical_site': anatomical_site,
                    'prediction_result': prediction_result,
                    'image_data': image_data_bytes,
                    'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'original_image_base64': original_image_base64,
                }
            else:
                return None
    except Exception as e:
        print(f"Error fetching last prediction: {e}")
        return None
    finally:
        if conn:
            conn.close()

def get_previous_predictions(user_id):
    # Check if the 'user_id' key exists in the session
    if 'user_id' not in session:
        # Handle case where the user is not authenticated
        return None

    conn = connect_db()
    try:
        with conn.cursor() as cursor:
            # Use parameterized query to fetch predictions
            sql = """
            SELECT id, anatomical_site, prediction_result, image_data, created_at, original_images
            FROM predictions
            WHERE user_id = %s
            AND created_at < (SELECT MAX(created_at) FROM predictions WHERE user_id = %s)
            """
            cursor.execute(sql, (user_id, user_id))
            previous_predictions = cursor.fetchall()

            # Extract the relevant fields from each tuple in the result
            formatted_predictions = []
            for prediction in previous_predictions:
                prediction_id, anatomical_site, prediction_result, image_data_bytes, created_at, original_image_bytes = prediction
                image_data = pickle.loads(image_data_bytes)
                formatted_prediction = {
                    'id':prediction_id,
                    'anatomical_site': anatomical_site,
                    'prediction_result': prediction_result,
                    'image_data': image_data,
                    'created_at': created_at.strftime('%Y-%m-%d %H:%M:%S'),
                    'original_images':original_image_bytes
                }
                formatted_predictions.append(formatted_prediction)
                
            return formatted_predictions
    except Exception as e:
        # Handle any database errors
        print(f"Error fetching previous predictions: {e}")
        return None
    finally:
        conn.close()


import numpy as np

def compare_skin_lesions(model, old_img_preprocessed, new_img_preprocessed, threshold=0.3):
    # Make predictions based only on the images
    prediction_old = model.predict(old_img_preprocessed)
    prediction_new = model.predict(new_img_preprocessed)
    
    # Calculate the difference between the predictions of the two images
    prediction_difference = np.abs(prediction_old - prediction_new)
    # Check if there is a positive evolution towards malignancy
    if np.any(prediction_difference > threshold):
        return "Noticed an evolution towards a suspicious lesion. It is recommended to consult a dermatologist for further evaluation. Warning: Please confirm, if the images are the same lesion"

    # Check if there is stability or negative evolution in the predictions
    else:
        return "Stability detected. Regular monitoring and consultation with a dermatologist are recommended. Visit your Skin AI App area regularly to monitor your lesions."

@app.route('/compare_predictions', methods=['GET', 'POST'])
def compare_predictions():
    # Verificar se o usuário está autenticado
    if 'user_id' not in session:
        # Redirecionar para a página de login se não estiver autenticado
        return redirect(url_for('login'))

    # Obter o ID do usuário da sessão
    user_id = session.get('user_id')

    if request.method == 'POST':
        # Verificar se o campo 'previous_prediction' está vazio
        selected_prediction_id = request.form.get('previous_prediction')
        if not selected_prediction_id:
            # Se nenhum ID de previsão selecionado for fornecido, renderizar uma mensagem de erro
            error_message = "Por favor, selecione uma previsão."
            return render_template('compare_predictions.html', error_message=error_message)
        
        # Guardar o ID da previsão selecionada na sessão
        session['selected_prediction_id'] = selected_prediction_id

        # Buscar previsões anteriores do banco de dados
        previous_predictions = get_previous_predictions(user_id)

        # Buscar detalhes da previsão selecionada do banco de dados
        selected_prediction = get_selected_prediction(selected_prediction_id)
        # Buscar a última previsão feita pelo usuário do banco de dados
        last_prediction = get_last_prediction(user_id)

        # Renderizar o modelo com os dados necessários
        return render_template('compare_predictions.html', selected_prediction=selected_prediction, last_prediction=last_prediction)
        
    else:
        # Buscar previsões anteriores do banco de dados para renderizar
        previous_predictions = get_previous_predictions(user_id)

        # Renderizar o modelo com os dados necessários
        return render_template('compare_predictions.html', previous_predictions=previous_predictions)



        

@app.route('/comparison_result', methods=['GET', 'POST'])
def comparison_result():
    # Verificar se o usuário está autenticado
    if 'user_id' not in session:
        # Redirecionar para a página de login se não estiver autenticado
        return redirect(url_for('login'))

    # Obter o ID do usuário da sessão
    user_id = session.get('user_id')

    prediction_id = session.get('selected_prediction_id')
    # Buscar detalhes da previsão selecionada do banco de dados
    selected_prediction = get_selected_prediction(prediction_id)
    # Buscar a última previsão feita pelo usuário do banco de dados
    last_prediction = get_last_prediction(user_id)

    original_selected_image_bytes = selected_prediction.get('original_image_base64')
    original_last_image_bytes = last_prediction.get('original_image_base64')
  
    # Carregar as imagens
    original_last_image = Image.open(io.BytesIO(base64.b64decode(original_last_image_bytes)))
    original_selected_image = Image.open(io.BytesIO(base64.b64decode(original_selected_image_bytes)))

    # Convert images to JPEG format in memory
    jpeg_image_last = io.BytesIO()
    original_last_image.save(jpeg_image_last, format='JPEG')
    jpeg_image_last.seek(0)

    jpeg_image_selected = io.BytesIO()
    original_selected_image.save(jpeg_image_selected, format='JPEG')
    jpeg_image_selected.seek(0)
    
    image_last= load_and_preprocess_image (jpeg_image_last)
    image_selected = load_and_preprocess_image (jpeg_image_selected)
    
    # Expanda a dimensão do lote das imagens pré-processadas
    old_img_preprocessed = np.expand_dims(image_selected, axis=0)
    new_img_preprocessed = np.expand_dims(image_last, axis=0)
    # Comparar as imagens de lesões cutâneas
    comparison_result = compare_skin_lesions(model_img, old_img_preprocessed, new_img_preprocessed, threshold=0.3)
    # Renderizar o template 'comparison_result.html' passando o resultado da comparação
    return render_template('comparison_result.html', comparison_result=comparison_result)


@app.route('/')
def home():
    if 'logged_in' in session:
        # Verifique se há um caminho da foto do perfil na sessão do usuário
        profile_photo_path = session.get('profile_photo_path')
        return render_template('index_en.html', username=session['username'], profile_photo_path=profile_photo_path)
    else:
        return redirect(url_for('index_en'))
    
# Route for the English index page
@app.route('/en')
def index_en():
    return render_template('index_en.html')

# Route for the Portuguese index page
@app.route('/pt')
def index_pt():
    return render_template('index_pt.html')

# Route for user registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'logged_in' in session:
        return redirect(url_for('user_area'))

    if request.method == 'POST':
        email = request.form['email']
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            error = 'Passwords do not match'
            return render_template('register.html', error=error)

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        # Connect to the database and insert the new user
        conn = connect_db()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO users (email, username, password) VALUES (%s, %s, %s)", (email, username, hashed_password))
            conn.commit()

            # Get the user ID of the newly registered user
            cursor.execute("SELECT user_id FROM users WHERE username = %s", (username,))
            user_id = cursor.fetchone()[0]

            # Initialize the session with the necessary information
            session['logged_in'] = True
            session['username'] = username
            session['user_id'] = user_id

            # Redirect to the user area
            return redirect(url_for('user_area'))

        except Error as e:
            error = f'Error registering user: {e}'
            return render_template('register.html', error=error)
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')

# Route for user login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'logged_in' in session:
        return redirect(url_for('user_area'))

    if request.method == 'POST':
        username_or_email = request.form['username_or_email']
        password = request.form['password']
        
        # Check if the provided field is an email or username
        is_email = '@' in username_or_email
        if is_email:
            query = "SELECT * FROM users WHERE email = %s"
        else:
            query = "SELECT * FROM users WHERE username = %s"

        # Check if the user exists in the database
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(query, (username_or_email,))
        user = cursor.fetchone()
        
        if user:
            # Check if the password hash matches the provided password
            if len(user) >= 4 and check_password_hash(user[3], password):
                # Set up user session upon successful login
                session['logged_in'] = True
                session['username'] = user[1]  # Assuming username is stored at index 1 in the user tuple
                session['user_id'] = user[0]  # Assuming user_id is stored at index 0 in the user tuple
                conn.close()
                return redirect(url_for('user_area'))
            else:
                error = 'Invalid username or password'
                conn.close()
                return render_template('login.html', error=error)
        else:
            error = 'Invalid username or email'
            conn.close()
            return render_template('login.html', error=error)
    
    return render_template('login.html')


@app.route('/upload_profile_photo', methods=['POST'])
def upload_profile_photo():
    # Verifique se o usuário está logado
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    # Verifique se a pasta de upload existe, crie-a se não existir
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    # Obtenha a imagem carregada do formulário
    profile_photo = request.files['profile_photo']

    # Leia a imagem como uma sequência de bytes
    image_bytes = profile_photo.read()

    # Redimensione a imagem
    resized_image = resize_image_photo(image_bytes)

    # Verifique se a imagem foi redimensionada com sucesso
    if resized_image is None:
        # Trate o caso em que a imagem não pode ser redimensionada
        return "Erro ao redimensionar a imagem", 400

    # Codifique a imagem redimensionada em base64
    encoded_image = base64.b64encode(resized_image)

    # Converta a imagem codificada em uma string
    encoded_image_str = encoded_image.decode('utf-8')

    # Verifique se 'user_id' está presente na sessão antes de acessá-lo
    user_id = None
    if 'user_id' in session:
        user_id = session['user_id']
    else:
        # Trate o caso em que 'user_id' não está presente na sessão (por exemplo, redirecione para a página de login)
        return redirect(url_for('login'))

    # Insira a imagem codificada no banco de dados
    insert_profile_photo_path_into_database(user_id, encoded_image_str)
    
    # Armazene os dados da imagem na sessão
    session['profile_photo'] = encoded_image_str
    print( encoded_image_str)

    # Retorne uma resposta de redirecionamento para a página 'user_area'
    return redirect(url_for('user_area'))

def insert_profile_photo_path_into_database(user_id, photo_path):
    conn = connect_db()
    cursor = conn.cursor()
    query = "UPDATE users SET profile_photo_path = %s WHERE user_id = %s"
    cursor.execute(query, (photo_path, user_id))
    conn.commit()
    conn.close()

def resize_image_photo(image_bytes, target_size=(50, 50)):
    try:
        # Decodifique a imagem para uma matriz OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Redimensiona a imagem para o tamanho alvo
        resized_img = cv2.resize(image, target_size)
        
        return resized_img
    except Exception as e:
        print("Erro ao redimensionar a imagem:", e)
        return None
import base64
import mysql.connector

def download_profile_photo(user_id):
    try:
        # Crie um cursor para executar consultas SQL
        conn = connect_db()
        cursor = conn.cursor()
        query = "SELECT profile_photo_path FROM users WHERE user_id = %s"
        cursor.execute(query, (user_id,))
        encoded_image = cursor.fetchone()[0]
        
        # Feche a conexão com o banco de dados
        cursor.close()
        conn.close()

        return encoded_image

    except Exception as e:
        print("Erro ao fazer download da imagem:", e)
        return None

    
# Route for password recovery
@app.route('/password_recovery', methods=['GET', 'POST'])
def password_recovery():
    if request.method == 'POST':
        username_or_email = request.form['username_or_email']
        is_email = '@' in username_or_email
        if is_email:
            query = "SELECT * FROM users WHERE email = %s"
        else:
            query = "SELECT * FROM users WHERE username = %s"
        
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute(query, (username_or_email,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if user:
            token = serializer.dumps(user[0], salt='recover-password')
            reset_url = url_for('reset_password', token=token, _external=True)
            msg = Message('Password Recovery - Skin Cancer App', recipients=[user[3]])
            msg.body = f"To reset your password, click the following link: {reset_url}"
            mail.send(msg)
            message = 'An email with instructions to reset your password has been sent to your email address.'
            return render_template('password_recovery.html', message=message)
        else:
            error = 'Invalid username or email'
            return render_template('password_recovery.html', error=error)
    
    return render_template('password_recovery.html')


# Route for password reset form
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        user_id = serializer.loads(token, salt='recover-password', max_age=3600)  # Token valid for 1 hour
    except SignatureExpired:
        flash('The password reset link has expired. Please try again.', 'error')
        return redirect(url_for('password_recovery'))
    except BadSignature:
        flash('Invalid password reset link. Please try again.', 'error')
        return redirect(url_for('password_recovery'))

    if request.method == 'POST':
        new_password = request.form['new_password']
        confirm_new_password = request.form['confirm_new_password']
        
        if new_password != confirm_new_password:
            flash('Passwords do not match', 'error')
            return render_template('reset_password.html', token=token)
        
        conn = connect_db()
        cursor = conn.cursor()
        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
        cursor.execute("UPDATE users SET password = %s WHERE id = %s", (hashed_password, user_id))
        conn.commit()
        cursor.close()
        conn.close()
        
        flash('Password reset successfully. You can now log in with your new password.', 'success')
        return redirect(url_for('login'))
    
    return render_template('reset_password.html', token=token)

# Route for the user area
@app.route('/user_area')
def user_area():
    if 'username' in session:
        username = session['username']
        user_id = session['user_id']
        
        # Chame a função para fazer o download da imagem da foto de perfil do usuário
        profile_photo = download_profile_photo(user_id)
        print (profile_photo)
        if profile_photo:
            # Se a imagem for baixada com sucesso, renderize o template com os dados da imagem
            return render_template('user_area.html', username=username, profile_photo=profile_photo)
        else:
            # Se houver algum erro ao baixar a imagem, renderize o template sem a imagem
            return render_template('user_area.html', username=username, profile_photo=None)
    else:
        return redirect(url_for('login'))



# Route for user logout
@app.route('/logout')
def logout():
    # Clear user session
    session.clear()
    # Redirect to login page (or any other appropriate page)
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True, use_debugger=False)
