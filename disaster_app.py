from disaster_app import app

#Run locally on computer
app.run(host='localhost', port=5000, debug=True)

#Run on the Udacity workspace
#app.run(host='0.0.0.0', port=3001, debug=True)