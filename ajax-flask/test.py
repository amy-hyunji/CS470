from flask import Flask, render_template, request
import json, hashlib

app = Flask(__name__)

@app.route('/test', methods = ['POST'])
def test():
    id = request.form['id']
    print(1)
    return json.dumps({'result': 'test'})

if __name__ == '__main__':
    app.run()
