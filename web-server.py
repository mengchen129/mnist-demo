from flask import Flask, render_template, request, make_response, jsonify
from werkzeug.utils import secure_filename

import keras_mnist_module as AI
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'upload/'

# 模板文件更新时自动刷新，无需重启服务器
app.config['TEMPLATES_AUTO_RELOAD'] = True

AI.init_network()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    print(request.args)
    file = request.files['file']
    print(file)
    # 保存上传的文件，用于测试验证
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))

    result = AI.predict(file)
    return make_response(jsonify({'code': 0, 'msg': 'ok', 'value': result}), 200)


if __name__ == '__main__':
    app.run()
