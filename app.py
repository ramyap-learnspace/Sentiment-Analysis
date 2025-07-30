from flask import Flask, render_template, request
import model
import traceback
import os
import pickle as pka

app = Flask(__name__)

# List of known valid user IDs
valid_userid = [
    '00sab00', '1234', 'zippy', 'zburt5', 'joshua', 'dorothy w',
    'rebecca', 'walker557', 'samantha', 'raeanne', 'kimmie',
    'cassie', 'moore222'
]

@app.route('/')
def view():
    return render_template('index.html')

# Safe loader for pickle files
def load_pickle(file_name):
    path = os.path.join('pickle_file', file_name)
    try:
        with open(path, 'rb') as f:
            return pka.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Pickle file '{file_name}' not found at {path}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to load pickle file '{file_name}': {e}")
        return None

@app.route('/recommend', methods=['POST'])
def recommend_top5():
    try:
        user_name = request.form.get('User Name', '').strip()
        print(f"Request method: {request.method}")
        print(f"User name received: {user_name}")

        if user_name in valid_userid:
            top20_products = model.recommend_products(user_name)
            if top20_products is None:
                return render_template('index.html', text='Recommendation system currently unavailable.')

            get_top5 = model.top5_products(top20_products)
            return render_template(
                'index.html',
                column_names=get_top5.columns.values,
                row_data=get_top5.values.tolist(),
                zip=zip,
                text='Recommended products'
            )
        else:
            return render_template('index.html', text='No recommendation found for the user.')

    except Exception as e:
        print("Error occurred:", e)
        traceback.print_exc()
        return render_template('index.html', text='An internal error occurred. Please try again later.')

if __name__ == '__main__':
    app.run(debug=True)
