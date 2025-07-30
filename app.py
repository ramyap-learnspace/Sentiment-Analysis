from flask import Flask, render_template, request
import traceback

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

@app.route('/recommend', methods=['POST'])
def recommend_top5():
    try:
        user_name = request.form.get('User Name', '').strip()
        print(f"Request method: {request.method}")
        print(f"User name received: {user_name}")

        if user_name in valid_userid:
            # ✅ Stubbed dummy product recommendation
            top5_products = [
                ["Product A"],
                ["Product B"],
                ["Product C"],
                ["Product D"],
                ["Product E"]
            ]
            column_names = ["Product Name"]

            return render_template(
                'index.html',
                column_names=column_names,
                row_data=top5_products,
                zip=zip,
                text='(Demo) Recommended products'
            )
        else:
            return render_template('index.html', text='No recommendation found for the user.')

    except Exception as e:
        print("Error occurred:", e)
        traceback.print_exc()
        return render_template('index.html', text='An internal error occurred. Please try again later.')

if __name__ == '__main__':
    app.run(debug=False)
