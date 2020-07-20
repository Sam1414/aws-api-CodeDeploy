from Load_Model_Locally import build, initialize
from flask import Flask, request

app = Flask(__name__)


@app.route('/url/')
def get_res():
    # Getting Input
    url = request.args.get('url')
    # Getting Result
    result = build(url)
    # Returning Result
    return result


@app.route('/')
def default_page():
    return '<html><body><h3>Enter URL in the following format:</h3><p><b>http://127.0.0.1:5000</b><i>/url/?url=</i><u>https://www.your_news_link.com</u></p></body></html>'


if __name__ == '__main__':
    initialize()
    app.debug = True
    app.run()
