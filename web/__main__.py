from www.app import create_app

app = create_app()
app.debug = True

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, use_reloader=False) 